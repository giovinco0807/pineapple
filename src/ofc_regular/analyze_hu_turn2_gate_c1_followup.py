"""Gate C1b/C1c follow-up for HU Turn2 calibration.

This is an evaluation-only analyzer. It autopsies Gate C1 false positives,
prepares high-MC recheck candidates, and designs repaired threshold candidates.
It does not start new teacher generation, T1 training, or production training.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .analyze_hu_turn2_gate_c1_large_calibration import (
    C1_GATE_THRESHOLD_GRID,
    C1_MIN_MARGIN_GRID,
    C1_REFERENCE_MARGIN_GRID,
    best_metric_row,
    baseline_confidence_group,
    sample_source_group,
    threshold_grid_rows,
    threshold_shortlist_rows,
)
from .analyze_hu_turn2_pilot_calibration import load_model, rows_for_split, write_csv
from .train_hu_turn2_pilot_model import load_cache, predict_all, target_matrix
from .train_torch_action_value import select_device

ANALYSIS_SPLITS = ("val", "test", "holdout")
PRIMARY_THRESHOLD_SPLIT = "test"
DEFAULT_FATAL_LOSS_LIMIT = 1.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("D:/ofc-pineapple-storage/regular-ofc-pineapple/feature_cache/hu_t2_gate_c1_5k_mc512"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/hu_turn2_stage8_pilot_2000_mc512_reference_override_cached_rank_wide.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/hu_turn2_stage1_pilot_training_5000_c1b_false_positive_autopsy"),
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--threshold-split", choices=("val", "test", "holdout"), default=PRIMARY_THRESHOLD_SPLIT)
    parser.add_argument("--target-min-states", type=int, default=5000)
    parser.add_argument("--target-max-states", type=int, default=10000)
    parser.add_argument("--high-mc-limit", type=int, default=120)
    parser.add_argument("--fatal-loss-limit", type=float, default=DEFAULT_FATAL_LOSS_LIMIT)
    return parser.parse_args()


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def margin_bucket_label(value: float, thresholds: tuple[float, ...], prefix: str) -> str:
    previous = None
    for threshold in thresholds:
        if value < threshold:
            if previous is None:
                return f"{prefix}_lt_{threshold:g}"
            return f"{prefix}_{previous:g}_to_{threshold:g}"
        previous = threshold
    return f"{prefix}_ge_{thresholds[-1]:g}"


def action_pattern(action: dict[str, Any] | None) -> str:
    if not action:
        return "unknown"
    placements = action.get("placements") or []
    rows = sorted(str(row) for _card, row in placements)
    discard_count = len(action.get("discards") or [])
    if not rows:
        return f"no_place_discard{discard_count}"
    return f"{'+'.join(rows)}_discard{discard_count}"


def action_text(action: dict[str, Any] | None) -> str:
    if not action:
        return ""
    placements = action.get("placements") or []
    placed = ",".join(f"{card}->{row}" for card, row in placements)
    discards = ",".join(str(card) for card in (action.get("discards") or []))
    return f"place[{placed}] discard[{discards}]"


def load_action_original_indices(cache_dir: Path, action_count: int) -> np.ndarray:
    path = cache_dir / "action_original_index.int16.mmap"
    if not path.exists():
        return np.arange(action_count, dtype=np.int16)
    return np.memmap(path, dtype=np.int16, mode="r", shape=(action_count,))


def split_name_for_state(cache: dict[str, Any], state_index: int) -> str:
    split_id = int(cache["split"][state_index])
    return {0: "train", 1: "val", 2: "test"}.get(split_id, "unknown")


def enriched_state_rows(
    cache: dict[str, Any],
    predictions: np.ndarray,
    targets: np.ndarray,
    action_original_indices: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offsets = cache["offsets"]
    for state_index, meta in enumerate(cache["state_metadata"]):
        start = int(offsets[state_index])
        end = int(offsets[state_index + 1])
        pred = predictions[start:end]
        y = targets[start:end]
        teacher_order = np.argsort(-y[:, 0], kind="mergesort")
        delta_order = np.argsort(-pred[:, 1], kind="mergesort")
        pred_order = np.argsort(-pred[:, 0], kind="mergesort")

        baseline = int(cache["baseline_action_index"][state_index])
        reference = int(cache["reference_action_index"][state_index])
        best = int(teacher_order[0])
        second = int(teacher_order[1]) if teacher_order.size > 1 else best
        candidate = int(delta_order[0])
        pred_best = int(pred_order[0])
        pred_second = int(pred_order[1]) if pred_order.size > 1 else pred_best

        absolute_candidate = start + candidate
        absolute_baseline = start + baseline
        candidate_se = safe_float(cache["action_ev_se"][absolute_candidate])
        baseline_se = safe_float(cache["action_ev_se"][absolute_baseline])
        gain_stderr_proxy = math.sqrt(candidate_se * candidate_se + baseline_se * baseline_se)
        actual_gain = float(y[candidate, 0] - y[baseline, 0])
        gate_logit = float(np.mean(pred[:, 4]))

        row = {
            "state_index": state_index,
            "split": split_name_for_state(cache, state_index),
            "bucket_group": meta.get("bucket_group", "unknown"),
            "run_bucket": meta.get("run_bucket", "unknown"),
            "source_bucket": meta.get("source_bucket", "unknown"),
            "source_group": "",
            "seat": meta.get("seat", "unknown"),
            "to_act_order": meta.get("to_act_order", "unknown"),
            "pilot_gate_label": meta.get("pilot_gate_label", "unknown"),
            "actual_high_regret": bool(meta.get("actual_high_regret", False)),
            "actual_low_margin": bool(meta.get("actual_low_margin", False)),
            "actual_teacher_disagreement": bool(meta.get("actual_teacher_disagreement", False)),
            "teacher_best_EV": float(y[best, 0]),
            "teacher_second_EV": float(y[second, 0]),
            "teacher_best_margin": float(y[best, 0] - y[second, 0]),
            "actual_delta_best_vs_baseline": float(y[best, 0] - y[baseline, 0]),
            "actual_delta_best_vs_reference": float(y[best, 0] - y[reference, 0]),
            "actual_delta_candidate_vs_baseline": actual_gain,
            "actual_delta_candidate_vs_reference": float(y[candidate, 0] - y[reference, 0]),
            "predicted_EV_best": float(pred[pred_best, 0]),
            "predicted_EV_second": float(pred[pred_second, 0]),
            "predicted_EV_margin_top1_top2": float(pred[pred_best, 0] - pred[pred_second, 0]),
            "predicted_delta_vs_baseline": float(pred[candidate, 1]),
            "predicted_delta_vs_reference": float(pred[candidate, 2]),
            "predicted_rank_score": float(pred[candidate, 3]),
            "reference_margin_raw": float(meta.get("baseline_model_margin", 0.0) or 0.0),
            "reference_margin_predicted": float(pred[pred_best, 0] - pred[baseline, 0]),
            "gate_logit": gate_logit,
            "gate_probability": sigmoid(gate_logit),
            "SE_delta": float(meta.get("SE_delta_best_vs_baseline", 0.0) or 0.0),
            "candidate_is_baseline": int(candidate == baseline),
            "candidate_false_positive": int(candidate != baseline and actual_gain < 0.0),
            "candidate_loss": max(0.0, -actual_gain),
            "action_count": int(end - start),
            "baseline_action_local_index": baseline,
            "reference_action_local_index": reference,
            "candidate_action_local_index": candidate,
            "teacher_best_action_local_index": best,
            "teacher_second_action_local_index": second,
            "baseline_action_original_index": int(action_original_indices[absolute_baseline]),
            "reference_action_original_index": int(action_original_indices[start + reference]),
            "candidate_action_original_index": int(action_original_indices[absolute_candidate]),
            "teacher_best_action_original_index": int(action_original_indices[start + best]),
            "teacher_second_action_original_index": int(action_original_indices[start + second]),
            "baseline_action_ev": float(y[baseline, 0]),
            "candidate_action_ev": float(y[candidate, 0]),
            "teacher_best_action_ev": float(y[best, 0]),
            "candidate_action_se": candidate_se,
            "baseline_action_se": baseline_se,
            "gain_stderr_proxy": gain_stderr_proxy,
            "gain_lcb_1p64": actual_gain - 1.64 * gain_stderr_proxy,
            "gain_lcb_1p96": actual_gain - 1.96 * gain_stderr_proxy,
        }
        row["source_group"] = sample_source_group(row)
        rows.append(row)
    return rows


def fires(row: dict[str, Any], *, min_margin: float, reference_min_margin: float, gate_threshold: float) -> bool:
    return (
        not int(row["candidate_is_baseline"])
        and float(row["predicted_delta_vs_baseline"]) >= min_margin
        and float(row["reference_margin_raw"]) >= reference_min_margin
        and float(row["gate_probability"]) >= gate_threshold
    )


def fired_for_config(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    subset = rows_for_split(rows, str(config["split"]))
    min_margin = float(config["hu_turn2_min_margin"])
    reference_min_margin = float(config["hu_turn2_reference_min_margin"])
    gate_threshold = float(config["gate_threshold"])
    return [
        row
        for row in subset
        if fires(row, min_margin=min_margin, reference_min_margin=reference_min_margin, gate_threshold=gate_threshold)
    ]


def config_label(config: dict[str, Any], prefix: str | None = None) -> str:
    base = (
        f"{config['split']}_m{float(config['hu_turn2_min_margin']):.2f}"
        f"_r{float(config['hu_turn2_reference_min_margin']):.2f}"
        f"_g{float(config['gate_threshold']):.2f}"
    )
    return f"{prefix}_{base}" if prefix else base


def select_analysis_configs(
    grid_rows: list[dict[str, Any]],
    shortlist_rows: list[dict[str, Any]],
    *,
    threshold_split: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, float, float, float]] = set()

    split_grid = [row for row in grid_rows if row["split"] == threshold_split]
    best = best_metric_row(split_grid)
    candidates: list[tuple[str, dict[str, Any]]] = []
    if best:
        candidates.append(("best_diagnostic", best))
    for row in shortlist_rows:
        candidates.append((f"shortlist_rank{int(row.get('rank', 0)):02d}", row))
    for row in split_grid:
        if int(row.get("override_count", 0)) > 0 and int(row.get("false_positive_count", 0)) > 0:
            candidates.append(("fp_grid", row))

    for prefix, row in candidates:
        key = (
            str(row["split"]),
            float(row["hu_turn2_min_margin"]),
            float(row["hu_turn2_reference_min_margin"]),
            float(row["gate_threshold"]),
        )
        if key in seen:
            continue
        seen.add(key)
        copied = dict(row)
        copied["analysis_label"] = config_label(copied, prefix)
        copied["is_primary_config"] = int(prefix == "best_diagnostic")
        selected.append(copied)
    return selected


def state_action_lookup_from_cache_inputs(cache: dict[str, Any], needed_state_indices: set[int]) -> dict[int, dict[str, Any]]:
    if not needed_state_indices:
        return {}
    input_files = cache["metadata"].get("input_files") or []
    if not input_files:
        return {}

    output: dict[int, dict[str, Any]] = {}
    state_index = 0
    for item in input_files:
        path = Path(str(item["path"]))
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                if state_index in needed_state_indices:
                    sample = json.loads(line)
                    output[state_index] = {
                        "actions": sample.get("actions") or [],
                        "baseline_action": sample.get("baseline_action"),
                        "reference_action": sample.get("reference_action"),
                        "fallback_action": sample.get("fallback_action"),
                        "state_id": sample.get("state_id"),
                        "sample_id": sample.get("sample_id"),
                        "hand_id": sample.get("hand_id"),
                    }
                    if len(output) == len(needed_state_indices):
                        return output
                state_index += 1
    return output


def action_by_local_index(lookup: dict[int, dict[str, Any]], state_index: int, local_index: int) -> dict[str, Any] | None:
    actions = lookup.get(state_index, {}).get("actions") or []
    if 0 <= local_index < len(actions):
        return actions[local_index]
    return None


def add_action_fields(row: dict[str, Any], lookup: dict[int, dict[str, Any]]) -> dict[str, Any]:
    state_index = int(row["state_index"])
    candidate = action_by_local_index(lookup, state_index, int(row["candidate_action_local_index"]))
    baseline = action_by_local_index(lookup, state_index, int(row["baseline_action_local_index"]))
    teacher = action_by_local_index(lookup, state_index, int(row["teacher_best_action_local_index"]))
    output = dict(row)
    for label, action in (("candidate", candidate), ("baseline", baseline), ("teacher_best", teacher)):
        output[f"{label}_action_pattern"] = action_pattern(action)
        output[f"{label}_action_text"] = action_text(action)
    sample = lookup.get(state_index, {})
    output["sample_id"] = sample.get("sample_id", "")
    output["hand_id"] = sample.get("hand_id", "")
    return output


def classify_false_positive(row: dict[str, Any]) -> str:
    gain = float(row["actual_delta_candidate_vs_baseline"])
    stderr = float(row.get("gain_stderr_proxy", 0.0) or 0.0)
    predicted = float(row["predicted_delta_vs_baseline"])
    reference = float(row["reference_margin_raw"])
    teacher_margin = float(row["teacher_best_margin"])
    if abs(gain) <= max(0.25, 1.96 * stderr):
        return "reference_or_teacher_noise"
    if reference < 0.25 or teacher_margin < 0.25:
        return "low_margin_ambiguous_spot"
    if sample_source_group(row) == "random_off_policy":
        return "source_specific_bias"
    if str(row.get("seat")) == "second":
        return "position_specific_bias"
    if predicted - gain >= 2.0:
        return "model_overestimation"
    return "model_overestimation"


def false_positive_event_rows(
    rows: list[dict[str, Any]],
    configs: list[dict[str, Any]],
    action_lookup: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for config in configs:
        for row in fired_for_config(rows, config):
            actual_gain = float(row["actual_delta_candidate_vs_baseline"])
            if actual_gain >= 0.0:
                continue
            enriched = add_action_fields(row, action_lookup)
            output.append(
                {
                    "analysis_label": config["analysis_label"],
                    "is_primary_config": int(config.get("is_primary_config", 0)),
                    "split": config["split"],
                    "hu_turn2_min_margin": config["hu_turn2_min_margin"],
                    "hu_turn2_reference_min_margin": config["hu_turn2_reference_min_margin"],
                    "gate_threshold": config["gate_threshold"],
                    "state_index": row["state_index"],
                    "sample_id": enriched.get("sample_id", ""),
                    "hand_id": enriched.get("hand_id", ""),
                    "source_group": sample_source_group(row),
                    "run_bucket": row.get("run_bucket", ""),
                    "bucket_group": row.get("bucket_group", ""),
                    "seat": row.get("seat", ""),
                    "baseline_confidence": baseline_confidence_group(row),
                    "predicted_delta_bucket": margin_bucket_label(float(row["predicted_delta_vs_baseline"]), (2.0, 2.5, 3.0, 5.0), "pred"),
                    "reference_margin_bucket": margin_bucket_label(float(row["reference_margin_raw"]), (0.1, 0.25, 0.5, 1.0), "ref"),
                    "gate_probability_bucket": margin_bucket_label(float(row["gate_probability"]), (0.6, 0.7, 0.8, 0.9), "gate"),
                    "predicted_delta_vs_baseline": row["predicted_delta_vs_baseline"],
                    "reference_margin_raw": row["reference_margin_raw"],
                    "gate_probability": row["gate_probability"],
                    "actual_delta_candidate_vs_baseline": actual_gain,
                    "candidate_loss": max(0.0, -actual_gain),
                    "gain_stderr_proxy": row["gain_stderr_proxy"],
                    "gain_lcb_1p64": row["gain_lcb_1p64"],
                    "gain_lcb_1p96": row["gain_lcb_1p96"],
                    "teacher_best_margin": row["teacher_best_margin"],
                    "action_count": row["action_count"],
                    "candidate_action_local_index": row["candidate_action_local_index"],
                    "baseline_action_local_index": row["baseline_action_local_index"],
                    "teacher_best_action_local_index": row["teacher_best_action_local_index"],
                    "candidate_action_original_index": row["candidate_action_original_index"],
                    "baseline_action_original_index": row["baseline_action_original_index"],
                    "teacher_best_action_original_index": row["teacher_best_action_original_index"],
                    "candidate_action_pattern": enriched.get("candidate_action_pattern", ""),
                    "baseline_action_pattern": enriched.get("baseline_action_pattern", ""),
                    "teacher_best_action_pattern": enriched.get("teacher_best_action_pattern", ""),
                    "candidate_action_text": enriched.get("candidate_action_text", ""),
                    "baseline_action_text": enriched.get("baseline_action_text", ""),
                    "teacher_best_action_text": enriched.get("teacher_best_action_text", ""),
                    "error_type": classify_false_positive(row),
                }
            )
    output.sort(key=lambda item: (-int(item["is_primary_config"]), -float(item["candidate_loss"]), item["analysis_label"]))
    return output


def group_rows(rows: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for key_values, items in sorted(grouped.items()):
        losses = [float(row.get("candidate_loss", 0.0) or 0.0) for row in items]
        gains = [float(row.get("actual_delta_candidate_vs_baseline", 0.0) or 0.0) for row in items]
        out = {key: value for key, value in zip(keys, key_values)}
        out.update(
            {
                "event_rows": len(items),
                "false_positive_rows": sum(1 for gain in gains if gain < 0.0),
                "false_positive_rate": sum(1 for gain in gains if gain < 0.0) / max(len(items), 1),
                "avg_gain": float(np.mean(gains)) if gains else 0.0,
                "avg_loss": float(np.mean(losses)) if losses else 0.0,
                "max_loss": max(losses) if losses else 0.0,
            }
        )
        output.append(out)
    return output


def fired_event_rows_for_configs(
    rows: list[dict[str, Any]],
    configs: list[dict[str, Any]],
    action_lookup: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for config in configs:
        for row in fired_for_config(rows, config):
            enriched = add_action_fields(row, action_lookup)
            actual_gain = float(row["actual_delta_candidate_vs_baseline"])
            false_positive = actual_gain < 0.0
            output.append(
                {
                    "analysis_label": config["analysis_label"],
                    "is_primary_config": int(config.get("is_primary_config", 0)),
                    "split": config["split"],
                    "hu_turn2_min_margin": config["hu_turn2_min_margin"],
                    "hu_turn2_reference_min_margin": config["hu_turn2_reference_min_margin"],
                    "gate_threshold": config["gate_threshold"],
                    "state_index": row["state_index"],
                    "sample_id": enriched.get("sample_id", ""),
                    "hand_id": enriched.get("hand_id", ""),
                    "source_group": sample_source_group(row),
                    "run_bucket": row.get("run_bucket", ""),
                    "bucket_group": row.get("bucket_group", ""),
                    "seat": row.get("seat", ""),
                    "baseline_confidence": baseline_confidence_group(row),
                    "predicted_delta_bucket": margin_bucket_label(float(row["predicted_delta_vs_baseline"]), (2.0, 2.5, 3.0, 5.0), "pred"),
                    "reference_margin_bucket": margin_bucket_label(float(row["reference_margin_raw"]), (0.1, 0.25, 0.5, 1.0), "ref"),
                    "gate_probability_bucket": margin_bucket_label(float(row["gate_probability"]), (0.6, 0.7, 0.8, 0.9), "gate"),
                    "predicted_delta_vs_baseline": row["predicted_delta_vs_baseline"],
                    "reference_margin_raw": row["reference_margin_raw"],
                    "gate_score": row["gate_probability"],
                    "teacher_gain": actual_gain,
                    "actual_delta_candidate_vs_baseline": actual_gain,
                    "false_positive_flag": int(false_positive),
                    "candidate_loss": row["candidate_loss"],
                    "gain_stderr_proxy": row["gain_stderr_proxy"],
                    "gain_lcb_1p64": row["gain_lcb_1p64"],
                    "gain_lcb_1p96": row["gain_lcb_1p96"],
                    "teacher_best_margin": row["teacher_best_margin"],
                    "candidate_action_local_index": row["candidate_action_local_index"],
                    "baseline_action_local_index": row["baseline_action_local_index"],
                    "teacher_best_action_local_index": row["teacher_best_action_local_index"],
                    "candidate_action_pattern": enriched.get("candidate_action_pattern", ""),
                    "baseline_action_pattern": enriched.get("baseline_action_pattern", ""),
                    "teacher_best_action_pattern": enriched.get("teacher_best_action_pattern", ""),
                    "candidate_action_text": enriched.get("candidate_action_text", ""),
                    "baseline_action_text": enriched.get("baseline_action_text", ""),
                    "teacher_best_action_text": enriched.get("teacher_best_action_text", ""),
                    "override_direction": (
                        f"{enriched.get('baseline_action_pattern', '')}->{enriched.get('candidate_action_pattern', '')}"
                    ),
                    "error_type": classify_false_positive(row) if false_positive else "",
                    "notes": "primary_best_diagnostic" if int(config.get("is_primary_config", 0)) else "additional_analysis_config",
                }
            )
    return output


def high_mc_recheck_candidate_rows(
    rows: list[dict[str, Any]],
    primary_config: dict[str, Any] | None,
    action_lookup: dict[int, dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    candidates: dict[tuple[int, int], dict[str, Any]] = {}

    def add(row: dict[str, Any], reason: str, priority: int) -> None:
        key = (int(row["state_index"]), int(row["candidate_action_local_index"]))
        current = candidates.get(key)
        if current is None or priority < int(current["priority"]):
            enriched = add_action_fields(row, action_lookup)
            requested_mc = 4096 if reason in {"false_positive", "near_threshold"} else 2048
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
                "accept_under_current_mc512_lcb_1p64": int(float(row["gain_lcb_1p64"]) > 0.0),
                "accept_under_current_mc512_lcb_1p96": int(float(row["gain_lcb_1p96"]) > 0.0),
                "candidate_action_local_index": row["candidate_action_local_index"],
                "baseline_action_local_index": row["baseline_action_local_index"],
                "teacher_best_action_local_index": row["teacher_best_action_local_index"],
                "candidate_action_pattern": enriched.get("candidate_action_pattern", ""),
                "baseline_action_pattern": enriched.get("baseline_action_pattern", ""),
                "teacher_best_action_pattern": enriched.get("teacher_best_action_pattern", ""),
                "candidate_action_text": enriched.get("candidate_action_text", ""),
                "baseline_action_text": enriched.get("baseline_action_text", ""),
                "teacher_best_action_text": enriched.get("teacher_best_action_text", ""),
                "current_mc_samples": 512,
                "requested_mc_samples": requested_mc,
                "high_mc_status": "planned_not_executed",
                "accept_rule": "accept only if high-MC gain_mean - 1.96 * stderr > 0",
            }

    test_rows = rows_for_split(rows, "test")
    if primary_config:
        primary_fired = fired_for_config(rows, primary_config)
        for row in primary_fired:
            if float(row["actual_delta_candidate_vs_baseline"]) < 0.0:
                add(row, "false_positive", 1)
        for row in sorted(primary_fired, key=lambda item: float(item["actual_delta_candidate_vs_baseline"]), reverse=True)[:25]:
            add(row, "primary_top_gain", 3)
        for row in primary_fired:
            add(row, "primary_override", 4)

        min_margin = float(primary_config["hu_turn2_min_margin"])
        reference_min_margin = float(primary_config["hu_turn2_reference_min_margin"])
        gate_threshold = float(primary_config["gate_threshold"])
        for row in test_rows:
            if int(row["candidate_is_baseline"]):
                continue
            close = (
                min_margin - 0.25 <= float(row["predicted_delta_vs_baseline"]) < min_margin + 0.25
                and float(row["reference_margin_raw"]) >= reference_min_margin
                and gate_threshold - 0.10 <= float(row["gate_probability"]) < gate_threshold + 0.10
            )
            if close:
                add(row, "near_threshold", 2)

    for row in sorted(test_rows, key=lambda item: float(item["predicted_delta_vs_baseline"]), reverse=True)[:50]:
        if not int(row["candidate_is_baseline"]):
            add(row, "top_predicted_gain", 5)

    output = sorted(
        candidates.values(),
        key=lambda item: (
            int(item["priority"]),
            float(item["gain_mean_minus_1p96_stderr_current_mc512"]),
            -float(item["predicted_delta_vs_baseline"]),
        ),
    )
    return output[:limit]


def high_mc_placeholder_result_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in candidates:
        output.append(
            {
                "state_index": row["state_index"],
                "sample_id": row.get("sample_id", ""),
                "hand_id": row.get("hand_id", ""),
                "reason": row["reason"],
                "source_group": row["source_group"],
                "seat": row["seat"],
                "current_mc_samples": row["current_mc_samples"],
                "requested_mc_samples": row["requested_mc_samples"],
                "result_status": "not_executed_no_event_replay_cli",
                "gain_mean": "",
                "gain_stderr": "",
                "lower_bound_90": "",
                "lower_bound_95": "",
                "sign_flip_count": "",
                "false_positive_after_high_mc": "",
                "still_positive_count": "",
                "still_negative_count": "",
                "current_mc512_gain_mean": row["teacher_gain_mean_current_mc512"],
                "current_mc512_gain_stderr_proxy": row["teacher_gain_stderr_proxy_current_mc512"],
                "current_mc512_lower_bound_90": row["gain_mean_minus_1p64_stderr_current_mc512"],
                "current_mc512_lower_bound_95": row["gain_mean_minus_1p96_stderr_current_mc512"],
            }
        )
    return output


def threshold_passes_strategy(row: dict[str, Any], strategy: dict[str, Any]) -> bool:
    if int(row["candidate_is_baseline"]):
        return False
    seat = str(row.get("seat", "unknown"))
    seat_thresholds = strategy.get("seat_thresholds", {})
    threshold = seat_thresholds.get(seat, strategy)
    if float(row["predicted_delta_vs_baseline"]) < float(threshold["min_margin"]):
        return False
    if float(row["reference_margin_raw"]) < float(threshold.get("reference_min_margin", 0.0)):
        return False
    if float(row["gate_probability"]) < float(threshold.get("gate_threshold", 0.0)):
        return False
    allowed_sources = strategy.get("allowed_source_groups")
    if allowed_sources and sample_source_group(row) not in set(allowed_sources):
        return False
    blocked_sources = strategy.get("blocked_source_groups")
    if blocked_sources and sample_source_group(row) in set(blocked_sources):
        return False
    if strategy.get("requires_current_lcb_1p64_positive") and float(row["gain_lcb_1p64"]) <= 0.0:
        return False
    if strategy.get("requires_current_lcb_1p96_positive") and float(row["gain_lcb_1p96"]) <= 0.0:
        return False
    return True


def repaired_strategy_specs() -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": "global_conservative_m3_r0_g0p9",
            "threshold_family": "global_conservative",
            "min_margin": 3.0,
            "reference_min_margin": 0.0,
            "gate_threshold": 0.9,
            "runtime_usable_without_teacher": True,
        },
        {
            "candidate_id": "global_conservative_m3_r0p5_g0p8",
            "threshold_family": "global_conservative",
            "min_margin": 3.0,
            "reference_min_margin": 0.5,
            "gate_threshold": 0.8,
            "runtime_usable_without_teacher": True,
        },
        {
            "candidate_id": "position_specific_first25_second30",
            "threshold_family": "position_specific",
            "min_margin": 2.5,
            "reference_min_margin": 0.0,
            "gate_threshold": 0.8,
            "seat_thresholds": {
                "first": {"min_margin": 2.5, "reference_min_margin": 0.0, "gate_threshold": 0.8},
                "second": {"min_margin": 3.0, "reference_min_margin": 0.0, "gate_threshold": 0.9},
            },
            "runtime_usable_without_teacher": True,
        },
        {
            "candidate_id": "source_filtered_no_random_m2p5_r0_g0p7",
            "threshold_family": "source_filtered",
            "min_margin": 2.5,
            "reference_min_margin": 0.0,
            "gate_threshold": 0.7,
            "blocked_source_groups": ["random_off_policy"],
            "runtime_usable_without_teacher": True,
        },
        {
            "candidate_id": "source_filtered_policy_near_m2p5_r0_g0p7",
            "threshold_family": "source_filtered",
            "min_margin": 2.5,
            "reference_min_margin": 0.0,
            "gate_threshold": 0.7,
            "allowed_source_groups": ["policy_on_distribution", "near_threshold_spots"],
            "runtime_usable_without_teacher": True,
        },
        {
            "candidate_id": "confidence_lcb164_m2p5_r0_g0p7",
            "threshold_family": "confidence_lower_bound_gated",
            "min_margin": 2.5,
            "reference_min_margin": 0.0,
            "gate_threshold": 0.7,
            "requires_current_lcb_1p64_positive": True,
            "runtime_usable_without_teacher": False,
        },
        {
            "candidate_id": "confidence_lcb196_m2p5_r0_g0p7",
            "threshold_family": "confidence_lower_bound_gated",
            "min_margin": 2.5,
            "reference_min_margin": 0.0,
            "gate_threshold": 0.7,
            "requires_current_lcb_1p96_positive": True,
            "runtime_usable_without_teacher": False,
        },
    ]


def repaired_threshold_candidate_rows(
    rows: list[dict[str, Any]],
    *,
    fatal_loss_limit: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for strategy in repaired_strategy_specs():
        for split_name in ANALYSIS_SPLITS:
            subset = rows_for_split(rows, split_name)
            fired = [row for row in subset if threshold_passes_strategy(row, strategy)]
            gains = [float(row["actual_delta_candidate_vs_baseline"]) for row in fired]
            losses = [max(0.0, -gain) for gain in gains]
            source_counts = Counter(sample_source_group(row) for row in fired)
            seat_counts = Counter(str(row.get("seat", "unknown")) for row in fired)
            lower_164 = [float(row["gain_lcb_1p64"]) for row in fired]
            lower_196 = [float(row["gain_lcb_1p96"]) for row in fired]
            false_positive_count = sum(1 for gain in gains if gain < 0.0)
            override_count = len(fired)
            fp_rate = false_positive_count / max(override_count, 1)
            avg_gain = float(np.mean(gains)) if gains else 0.0
            max_loss = max(losses) if losses else 0.0
            dominant_source_share = max(source_counts.values(), default=0) / max(override_count, 1)
            dominant_seat_share = max(seat_counts.values(), default=0) / max(override_count, 1)
            lower_bound_positive = bool(lower_196 and min(lower_196) > 0.0)
            c2_small_ready = (
                split_name == PRIMARY_THRESHOLD_SPLIT
                and override_count >= 30
                and fp_rate <= 0.10
                and avg_gain > 0.0
                and lower_bound_positive
                and max_loss <= fatal_loss_limit
                and dominant_source_share < 0.80
                and dominant_seat_share < 0.80
            )
            reasons: list[str] = []
            if override_count < 30:
                reasons.append("override_count_lt_30")
            if fp_rate > 0.10:
                reasons.append("false_positive_rate_gt_10pct")
            if avg_gain <= 0.0:
                reasons.append("avg_gain_not_positive")
            if not lower_bound_positive:
                reasons.append("current_mc512_lcb196_not_all_positive")
            if max_loss > fatal_loss_limit:
                reasons.append("worst_loss_above_limit")
            if dominant_source_share >= 0.80:
                reasons.append("source_biased")
            if dominant_seat_share >= 0.80:
                reasons.append("position_biased")
            if not strategy.get("runtime_usable_without_teacher", True):
                reasons.append("requires_high_mc_teacher_lcb")
            output.append(
                {
                    "candidate_id": strategy["candidate_id"],
                    "threshold_family": strategy["threshold_family"],
                    "split": split_name,
                    "runtime_usable_without_teacher": int(bool(strategy.get("runtime_usable_without_teacher", True))),
                    "evaluated_states": len(subset),
                    "override_count": override_count,
                    "override_rate": override_count / max(len(subset), 1),
                    "teacher_avg_gain_on_override": avg_gain,
                    "median_gain_on_override": float(np.median(gains)) if gains else 0.0,
                    "false_positive_count": false_positive_count,
                    "false_positive_rate": fp_rate,
                    "max_loss": max_loss,
                    "gain_lcb_1p64_min": min(lower_164) if lower_164 else 0.0,
                    "gain_lcb_1p96_min": min(lower_196) if lower_196 else 0.0,
                    "lcb_1p64_positive_count": sum(1 for value in lower_164 if value > 0.0),
                    "lcb_1p96_positive_count": sum(1 for value in lower_196 if value > 0.0),
                    "source_counts": json.dumps(dict(source_counts), sort_keys=True),
                    "seat_counts": json.dumps(dict(seat_counts), sort_keys=True),
                    "dominant_source_share": dominant_source_share,
                    "dominant_seat_share": dominant_seat_share,
                    "source_biased": int(override_count > 0 and dominant_source_share >= 0.80),
                    "position_biased": int(override_count > 0 and dominant_seat_share >= 0.80),
                    "c2_small_ready": int(c2_small_ready),
                    "c2_small_blockers": ";".join(reasons),
                }
            )
    return output


def c2_small_candidate_threshold_rows(repaired_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in repaired_rows:
        if row.get("split") != PRIMARY_THRESHOLD_SPLIT:
            continue
        blockers = str(row.get("c2_small_blockers", ""))
        status = "go" if int(row.get("c2_small_ready", 0)) else "no_go"
        if status == "no_go" and int(row.get("override_count", 0)) > 0 and float(row.get("false_positive_rate", 0.0)) <= 0.10:
            status = "conditional_pending_more_events_and_high_mc"
        output.append(
            {
                "candidate_id": row["candidate_id"],
                "threshold_family": row["threshold_family"],
                "candidate_status": status,
                "c2_small_ready": row["c2_small_ready"],
                "override_count": row["override_count"],
                "false_positive_rate": row["false_positive_rate"],
                "teacher_avg_gain_on_override": row["teacher_avg_gain_on_override"],
                "max_loss": row["max_loss"],
                "gain_lcb_1p64_min": row["gain_lcb_1p64_min"],
                "gain_lcb_1p96_min": row["gain_lcb_1p96_min"],
                "source_counts": row["source_counts"],
                "seat_counts": row["seat_counts"],
                "scope_note": "fixed threshold must be rerun on separate-seed C2-small heldout before any scale-up",
                "blockers": blockers,
            }
        )
    output.sort(
        key=lambda item: (
            {"go": 0, "conditional_pending_more_events_and_high_mc": 1, "no_go": 2}.get(str(item["candidate_status"]), 3),
            -float(item["teacher_avg_gain_on_override"]),
            float(item["false_positive_rate"]),
            -int(item["override_count"]),
        )
    )
    return output


def write_false_positive_autopsy(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    configs: list[dict[str, Any]],
    false_positive_rows: list[dict[str, Any]],
    fired_rows: list[dict[str, Any]],
    high_mc_rows: list[dict[str, Any]],
) -> None:
    primary = next((config for config in configs if int(config.get("is_primary_config", 0))), configs[0] if configs else None)
    primary_fired = [row for row in fired_rows if primary and row["analysis_label"] == primary["analysis_label"]]
    primary_fp = [row for row in false_positive_rows if primary and row["analysis_label"] == primary["analysis_label"]]
    fp_errors = Counter(str(row.get("error_type", "unknown")) for row in primary_fp)
    fp_sources = Counter(str(row.get("source_group", "unknown")) for row in primary_fp)
    fp_seats = Counter(str(row.get("seat", "unknown")) for row in primary_fp)
    lines = [
        "# HU Turn2 Gate C1b False Positive Autopsy",
        "",
        "This follow-up is evaluation-only. It does not authorize production training, T1 training, or a 50k teacher run.",
        "",
        "## Primary C1 Row",
        "",
    ]
    if primary:
        primary_fp_rate = len(primary_fp) / max(len(primary_fired), 1)
        avg_gain = float(np.mean([float(row["actual_delta_candidate_vs_baseline"]) for row in primary_fired])) if primary_fired else 0.0
        lines.extend(
            [
                f"- config: `{primary['analysis_label']}`",
                f"- override count: `{len(primary_fired)}`",
                f"- false positives: `{len(primary_fp)}`",
                f"- false positive rate: `{primary_fp_rate:.4f}`",
                f"- teacher avg gain on override: `{avg_gain:.4f}`",
                f"- C2-ready: `No`",
                "",
            ]
        )
    lines.extend(
        [
            "## False Positive Axes",
            "",
            f"- by error type: `{json.dumps(dict(fp_errors), sort_keys=True)}`",
            f"- by source: `{json.dumps(dict(fp_sources), sort_keys=True)}`",
            f"- by position: `{json.dumps(dict(fp_seats), sort_keys=True)}`",
            "",
            "## Interpretation",
            "",
            "- The current best diagnostic row is not C2-ready because override count is below 30 and false positive rate is above 10%.",
            "- False positives are retained with source, position, margin bucket, action pattern, and current MC512 lower-bound proxies for targeted recheck.",
            "- High-MC recheck is planned, not executed, by this script. Use the generated candidate list for MC2048/MC4096 only.",
            "",
            "## Generated Evidence",
            "",
            f"- total loaded states: `{len(rows)}`",
            f"- selected analysis configs: `{len(configs)}`",
            f"- false-positive config/event rows: `{len(false_positive_rows)}`",
            f"- high-MC recheck candidates: `{len(high_mc_rows)}`",
            "",
            "## Guardrails",
            "",
            "- Production training remains `No-Go`.",
            "- 50k teacher remains `No-Go/Pending` until a repaired threshold passes heldout C2 criteria.",
            "- C1c candidates must be accepted only when high-MC lower bounds are positive.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_go_nogo_for_c2_small(
    path: Path,
    *,
    repaired_rows: list[dict[str, Any]],
    high_mc_rows: list[dict[str, Any]],
) -> None:
    ready = [row for row in repaired_rows if int(row.get("c2_small_ready", 0))]
    lines = [
        "# Gate C1c Go/No-Go For C2 Small",
        "",
        "- production candidate training: `No-Go`",
        "- T1/main training: `No-Go`",
        "- 50k teacher: `No-Go/Pending`",
        "- C2-small: `No-Go`",
        "",
        "## C2-Ready Criteria",
        "",
        "- minimum override count >= 30",
        "- false positive rate <= 10%, preferred <= 5%",
        "- teacher avg gain positive",
        "- high-MC lower bound positive",
        "- worst override not fatal",
        "- source/position bias scoped or limited",
        "- C2 heldout must use a separate seed",
        "",
        "## Current Result",
        "",
        f"- repaired threshold rows evaluated: `{len(repaired_rows)}`",
        f"- C2-small ready rows: `{len(ready)}`",
        f"- high-MC recheck candidates generated: `{len(high_mc_rows)}`",
        "- high-MC recheck executed: `No`",
        "",
        "## Decision",
        "",
        "C2-small remains `No-Go`. The next safe action is to run MC2048/MC4096 only for `high_mc_recheck_candidates.csv`, then rerun this report with confirmed lower bounds.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_calibration_c1b_summary(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    configs: list[dict[str, Any]],
    false_positive_rows: list[dict[str, Any]],
    fired_rows: list[dict[str, Any]],
    high_mc_rows: list[dict[str, Any]],
    repaired_rows: list[dict[str, Any]],
) -> None:
    primary = next((config for config in configs if int(config.get("is_primary_config", 0))), configs[0] if configs else None)
    primary_fired = [row for row in fired_rows if primary and row["analysis_label"] == primary["analysis_label"]]
    primary_fp = [row for row in false_positive_rows if primary and row["analysis_label"] == primary["analysis_label"]]
    test_repaired = [row for row in repaired_rows if row.get("split") == PRIMARY_THRESHOLD_SPLIT]
    conditional = [
        row
        for row in test_repaired
        if int(row.get("override_count", 0)) > 0
        and float(row.get("false_positive_rate", 0.0)) <= 0.10
        and float(row.get("teacher_avg_gain_on_override", 0.0)) > 0.0
    ]
    lines = [
        "# HU Turn2 Gate C1b/C1c Summary",
        "",
        "## Scope",
        "",
        "- Gate: `C1b/C1c`",
        "- Purpose: false-positive autopsy, high-MC recheck preparation, repaired threshold design",
        "- Production training: `No-Go`",
        "- T1 training: `No-Go`",
        "- 50k teacher: `No-Go/Pending`",
        "",
        "## Primary Diagnostic",
        "",
    ]
    if primary:
        fp_rate = len(primary_fp) / max(len(primary_fired), 1)
        avg_gain = float(np.mean([float(row["teacher_gain"]) for row in primary_fired])) if primary_fired else 0.0
        lines.extend(
            [
                f"- config: `{primary['analysis_label']}`",
                f"- override count: `{len(primary_fired)}`",
                f"- false positives: `{len(primary_fp)}`",
                f"- false positive rate: `{fp_rate:.4f}`",
                f"- teacher avg gain: `{avg_gain:.4f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## C1b Findings",
            "",
            f"- false-positive event rows across analysis configs: `{len(false_positive_rows)}`",
            f"- primary false-positive sources: `{json.dumps(dict(Counter(row['source_group'] for row in primary_fp)), sort_keys=True)}`",
            f"- primary false-positive positions: `{json.dumps(dict(Counter(row['seat'] for row in primary_fp)), sort_keys=True)}`",
            f"- primary false-positive error types: `{json.dumps(dict(Counter(row['error_type'] for row in primary_fp)), sort_keys=True)}`",
            "",
            "## C1c High-MC",
            "",
            f"- high-MC candidates: `{len(high_mc_rows)}`",
            "- high-MC execution: `not_executed_no_event_replay_cli`",
            "- reason: existing HU Turn2 teacher CLI samples/generates buckets; it does not replay arbitrary event rows with identical state/action identity.",
            "",
            "## Repaired Thresholds",
            "",
            f"- repaired threshold rows: `{len(repaired_rows)}`",
            f"- test split conditional rows: `{len(conditional)}`",
            f"- C2-small ready rows: `{sum(1 for row in repaired_rows if int(row.get('c2_small_ready', 0)))}`",
            "",
            "## Decision",
            "",
            "C2-small is `No-Go` from this C1 dataset. The only defensible next step is targeted high-MC replay support for the generated candidate rows, then a separate-seed C2-small heldout with fixed thresholds.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_high_mc_command_plan(path: Path, *, high_mc_rows: list[dict[str, Any]]) -> None:
    requested = Counter(str(row.get("requested_mc_samples", "")) for row in high_mc_rows)
    lines = [
        "# High-MC Recheck Command Plan",
        "",
        "High-MC was not executed in this pass because the current HU Turn2 teacher CLI does not expose an event replay mode.",
        "",
        "## Required runner",
        "",
        "Add or use a runner that reads `high_mc_recheck_candidates.csv`, reconstructs the exact saved state, evaluates the baseline/candidate/teacher actions with common random futures, and writes `high_mc_recheck_results.csv`.",
        "",
        "Required output columns:",
        "",
        "- `gain_mean`",
        "- `gain_stderr`",
        "- `lower_bound_90 = gain_mean - 1.64 * stderr`",
        "- `lower_bound_95 = gain_mean - 1.96 * stderr`",
        "- `sign_flip_count`",
        "- `false_positive_after_high_mc`",
        "- `still_positive_count`",
        "- `still_negative_count`",
        "",
        "## Candidate counts",
        "",
        f"- total candidates: `{len(high_mc_rows)}`",
        f"- requested MC buckets: `{json.dumps(dict(requested), sort_keys=True)}`",
        "",
        "## Guardrail",
        "",
        "Do not run broad 50k teacher from this list. Recheck only these event rows first.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started_at = time.time()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    cache_dir = args.cache_dir.resolve()
    device = select_device(torch, args.device)
    cache = load_cache(cache_dir)
    targets = target_matrix(cache)
    action_original_indices = load_action_original_indices(cache_dir, int(cache["metadata"]["action_count"]))
    net, stats, model_payload = load_model(torch, args.model.resolve(), device)
    predictions = predict_all(torch, net, cache, stats, device, args.batch_size)
    rows = enriched_state_rows(cache, predictions, targets, action_original_indices)

    sample_status = "target_met" if len(rows) >= args.target_min_states else "pilot_replay_only_insufficient_sample"
    grid_rows = threshold_grid_rows(
        rows,
        sample_status=sample_status,
        target_min_states=args.target_min_states,
        target_max_states=args.target_max_states,
    )
    shortlist = threshold_shortlist_rows(grid_rows, threshold_split=args.threshold_split, sample_status=sample_status)
    configs = select_analysis_configs(grid_rows, shortlist, threshold_split=args.threshold_split)

    needed_state_indices: set[int] = set()
    for config in configs:
        for row in fired_for_config(rows, config):
            needed_state_indices.add(int(row["state_index"]))
    action_lookup = state_action_lookup_from_cache_inputs(cache, needed_state_indices)

    false_positive_rows = false_positive_event_rows(rows, configs, action_lookup)
    all_fired_rows = fired_event_rows_for_configs(rows, configs, action_lookup)
    primary_config = next((config for config in configs if int(config.get("is_primary_config", 0))), configs[0] if configs else None)
    high_mc_rows = high_mc_recheck_candidate_rows(rows, primary_config, action_lookup, limit=args.high_mc_limit)
    repaired_rows = repaired_threshold_candidate_rows(rows, fatal_loss_limit=args.fatal_loss_limit)

    write_csv(output_dir / "false_positive_events.csv", false_positive_rows)
    write_csv(output_dir / "override_candidate_events.csv", all_fired_rows)
    write_csv(
        output_dir / "fp_by_source_position.csv",
        group_rows(all_fired_rows, ("analysis_label", "source_group", "seat", "baseline_confidence")),
    )
    write_csv(
        output_dir / "fp_by_margin_bucket.csv",
        group_rows(all_fired_rows, ("analysis_label", "predicted_delta_bucket", "reference_margin_bucket", "gate_probability_bucket")),
    )
    write_csv(
        output_dir / "fp_action_type_breakdown.csv",
        group_rows(all_fired_rows, ("analysis_label", "candidate_action_pattern", "baseline_action_pattern", "teacher_best_action_pattern")),
    )
    write_csv(output_dir / "high_mc_recheck_candidates.csv", high_mc_rows)
    write_csv(output_dir / "high_mc_recheck_results.csv", high_mc_placeholder_result_rows(high_mc_rows))
    write_csv(output_dir / "repaired_threshold_candidates.csv", repaired_rows)
    write_csv(output_dir / "c2_small_candidate_thresholds.csv", c2_small_candidate_threshold_rows(repaired_rows))

    write_false_positive_autopsy(
        output_dir / "false_positive_autopsy.md",
        rows=rows,
        configs=configs,
        false_positive_rows=false_positive_rows,
        fired_rows=all_fired_rows,
        high_mc_rows=high_mc_rows,
    )
    write_go_nogo_for_c2_small(output_dir / "go_nogo_for_c2_small.md", repaired_rows=repaired_rows, high_mc_rows=high_mc_rows)
    write_calibration_c1b_summary(
        output_dir / "calibration_c1b_summary.md",
        rows=rows,
        configs=configs,
        false_positive_rows=false_positive_rows,
        fired_rows=all_fired_rows,
        high_mc_rows=high_mc_rows,
        repaired_rows=repaired_rows,
    )
    write_high_mc_command_plan(output_dir / "high_mc_recheck_command_plan.md", high_mc_rows=high_mc_rows)

    manifest = {
        "schema": "hu_turn2_gate_c1_followup_v1",
        "cache_dir": str(cache_dir),
        "model": str(args.model.resolve()),
        "output_dir": str(output_dir),
        "state_count": len(rows),
        "action_count": int(cache["metadata"]["action_count"]),
        "sample_status": sample_status,
        "threshold_split": args.threshold_split,
        "analysis_config_count": len(configs),
        "false_positive_event_rows": len(false_positive_rows),
        "high_mc_recheck_candidates": len(high_mc_rows),
        "repaired_threshold_rows": len(repaired_rows),
        "c2_small_ready_rows": sum(1 for row in repaired_rows if int(row.get("c2_small_ready", 0))),
        "high_mc_executed": False,
        "production_training": "No-Go",
        "teacher_50k": "No-Go/Pending",
        "model_payload": {
            "model_kind": model_payload.get("model_kind"),
            "feature_dim": model_payload.get("feature_dim"),
            "hidden_layer_sizes": model_payload.get("hidden_layer_sizes"),
        },
        "elapsed_seconds": time.time() - started_at,
    }
    (output_dir / "gate_c1_followup_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
