"""Audit HU T2 Stage8c whole-game risk-head predictions.

This is an analysis tool for the Stage8c risk-head smoke model. It keeps the
same label boundary as the trainer:

- positive: whole-game risk-only realized losses
- negative: whole-game non-loss controls

It does not approve runtime use. Its main purpose is to show whether the risk
head generalizes beyond the training split and whether any threshold is useful
enough to justify another runtime experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .train_hu_turn2_stage8c_risk_head import average_precision, roc_auc

DEFAULT_PREDICTIONS = Path("outputs/training/hu_turn2_stage8c_risk_head_smoke/risk_head_predictions.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/training/hu_turn2_stage8c_risk_head_smoke_audit")
DEFAULT_THRESHOLDS = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
DEFAULT_RAW_SCORE_QUANTILES = (0.80, 0.85, 0.90, 0.95, 0.98)
RAW_SCORE_NAMES = (
    "risk_probability",
    "predicted_delta",
    "negative_predicted_delta",
    "abs_predicted_delta",
    "gate_probability",
    "confirm_delta",
    "negative_confirm_delta",
    "confirm_delta_z",
    "negative_confirm_delta_z",
    "confirm_minus_predicted_delta",
    "candidate_rank",
    "negative_candidate_rank",
    "candidate_rank_inverse",
)
RAW_SCORE_PRIMARY_METRIC_SOURCE = "realized_veto_utility_on_holdout_labels"
RAW_CONFIRM_SCORE_ROLE = "runtime_field_diagnostic_only"
RAW_CONFIRM_SCORE_PERFORMANCE_CLAIM_ALLOWED = False


def raw_score_metric_role(score_name: str) -> str:
    if "confirm_delta" in score_name:
        return RAW_CONFIRM_SCORE_ROLE
    if score_name == "risk_probability":
        return "trained_risk_head_score"
    return "runtime_proxy_feature_score"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--thresholds",
        default=",".join(str(value) for value in DEFAULT_THRESHOLDS),
        help="Comma-separated risk_probability thresholds to sweep.",
    )
    parser.add_argument("--deciles", type=int, default=10)
    parser.add_argument("--top-errors", type=int, default=50)
    parser.add_argument(
        "--runtime-group-min-selected",
        type=int,
        default=10,
        help="Minimum selected rows in both val and test for a runtime group threshold to be marked stable.",
    )
    parser.add_argument(
        "--runtime-group-target-selected",
        type=int,
        default=50,
        help="Target selected rows per holdout split used to estimate fresh-heldout sizing for near-miss runtime groups.",
    )
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_thresholds(value: str) -> list[float]:
    thresholds = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not thresholds:
        raise ValueError("at least one threshold is required")
    return sorted(set(thresholds))


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def label_array(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([safe_int(row.get("label")) for row in rows], dtype=np.float32)


def infer_target_mode(rows: list[dict[str, Any]]) -> str:
    groups = {str(row.get("risk_target_group") or row.get("recommended_training_use") or "") for row in rows}
    if any(group.startswith("local_ev_") for group in groups):
        return "local_ev_negative"
    if {"whole_game_risk_only", "whole_game_non_loss_control"} & groups:
        return "whole_game_risk"
    return "unknown"


def probability_array(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([safe_float(row.get("risk_probability")) for row in rows], dtype=np.float32)


def realized_delta_array(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([safe_float(row.get("realized_delta")) for row in rows], dtype=np.float32)


def realized_loss_array(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.maximum(0.0, -realized_delta_array(rows))


def raw_score_value(row: dict[str, Any], score_name: str) -> float:
    predicted_delta = safe_float(row.get("predicted_delta"))
    gate_probability = safe_float(row.get("gate_probability"))
    confirm_delta = safe_float(row.get("confirm_delta"))
    confirm_delta_se = max(safe_float(row.get("confirm_delta_se")), 1e-6)
    candidate_rank = float(max(1, min(safe_int(row.get("candidate_ev_rank"), 9999), 9999)))
    if score_name == "risk_probability":
        return safe_float(row.get("risk_probability"))
    if score_name == "predicted_delta":
        return predicted_delta
    if score_name == "negative_predicted_delta":
        return -predicted_delta
    if score_name == "abs_predicted_delta":
        return abs(predicted_delta)
    if score_name == "gate_probability":
        return gate_probability
    if score_name == "confirm_delta":
        return confirm_delta
    if score_name == "negative_confirm_delta":
        return -confirm_delta
    if score_name == "confirm_delta_z":
        return confirm_delta / confirm_delta_se
    if score_name == "negative_confirm_delta_z":
        return -(confirm_delta / confirm_delta_se)
    if score_name == "confirm_minus_predicted_delta":
        return confirm_delta - predicted_delta
    if score_name == "candidate_rank":
        return candidate_rank
    if score_name == "negative_candidate_rank":
        return -candidate_rank
    if score_name == "candidate_rank_inverse":
        return 1.0 / candidate_rank
    raise ValueError(f"unknown raw score {score_name!r}")


def raw_score_array(rows: list[dict[str, Any]], score_name: str) -> np.ndarray:
    return np.asarray([raw_score_value(row, score_name) for row in rows], dtype=np.float32)


def source_family(row: dict[str, Any]) -> str:
    source = str(row.get("source_log") or "")
    lower = source.lower()
    run = source_run(row).lower()
    if "riskplus2" in run:
        return "scd1_riskplus2"
    if "riskplus" in run:
        return "scd1_riskplus"
    if "500fires" in run:
        return "scd1_500fires"
    if "risk-expand" in lower:
        return "risk_expand"
    if "risk-fill" in lower:
        return "risk_fill"
    if "risk-second" in lower:
        return "risk_second"
    if "risk_control_extraction_smoke" in lower or "risk-control" in lower:
        return "risk_control_smoke"
    if not source:
        return "unknown"
    return "other"


def source_run(row: dict[str, Any]) -> str:
    source = str(row.get("source_log") or "")
    if not source:
        return "unknown"
    parts = [part for part in re.split(r"[\\/]+", source) if part]
    for marker in ("gcp_runs", "evals", "training"):
        if marker in parts:
            index = parts.index(marker)
            if index + 1 < len(parts):
                return parts[index + 1]
    return "other"


def source_seed(row: dict[str, Any]) -> str:
    source = str(row.get("source_log") or "")
    match = re.search(r"seed(\d+)", source)
    if match:
        return match.group(1)
    hand_seed = str(row.get("hand_seed") or "")
    if len(hand_seed) >= 10 and hand_seed[:10].isdigit():
        return hand_seed[:10]
    return "unknown"


def rank_bucket(row: dict[str, Any]) -> str:
    rank = safe_int(row.get("candidate_ev_rank"), 9999)
    if rank <= 1:
        return "rank_1"
    if rank <= 3:
        return "rank_2_3"
    if rank <= 5:
        return "rank_4_5"
    if rank < 9999:
        return "rank_6_plus"
    return "rank_unknown"


def split_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {"all": list(rows)}
    for split in sorted({str(row.get("split") or "") for row in rows}):
        if split:
            groups[split] = [row for row in rows if str(row.get("split") or "") == split]
    return groups


def subset_metrics(rows: list[dict[str, Any]], *, group_field: str, group_value: str) -> dict[str, Any]:
    labels = label_array(rows)
    probabilities = probability_array(rows)
    realized_delta = realized_delta_array(rows)
    realized_loss = np.maximum(0.0, -realized_delta)
    if labels.size == 0:
        return {
            "group_field": group_field,
            "group_value": group_value,
            "rows": 0,
        }
    return {
        "group_field": group_field,
        "group_value": group_value,
        "rows": int(labels.size),
        "positives": int(np.sum(labels > 0.5)),
        "positive_rate": float(np.mean(labels)),
        "risk_probability_mean": float(np.mean(probabilities)),
        "risk_probability_p90": float(np.quantile(probabilities, 0.90)),
        "risk_probability_p95": float(np.quantile(probabilities, 0.95)),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "average_precision": average_precision(labels, probabilities),
        "roc_auc": roc_auc(labels, probabilities),
        "realized_delta_mean": float(np.mean(realized_delta)),
        "realized_loss_sum": float(np.sum(realized_loss)),
        "realized_loss_mean": float(np.mean(realized_loss)),
    }


def split_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"split": name, **{key: value for key, value in subset_metrics(group, group_field="split", group_value=name).items() if key not in {"group_field", "group_value"}}}
        for name, group in split_groups(rows).items()
    ]


def _threshold_metrics_for_group(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    labels = label_array(rows) > 0.5
    probabilities = probability_array(rows)
    realized_delta = realized_delta_array(rows)
    realized_loss = np.maximum(0.0, -realized_delta)
    total_loss = float(np.sum(realized_loss))
    selected = probabilities >= float(threshold)
    tp = int(np.sum(selected & labels))
    fp = int(np.sum(selected & ~labels))
    tn = int(np.sum(~selected & ~labels))
    fn = int(np.sum(~selected & labels))
    selected_count = int(np.sum(selected))
    selected_loss = float(np.sum(realized_loss[selected])) if selected_count else 0.0
    selected_gain = float(np.sum(np.maximum(0.0, realized_delta[selected]))) if selected_count else 0.0
    selected_delta_sum = float(np.sum(realized_delta[selected])) if selected_count else 0.0
    selected_delta = float(np.mean(realized_delta[selected])) if selected_count else 0.0
    veto_net_gain = -selected_delta_sum
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "threshold": float(threshold),
        "rows": int(labels.size),
        "selected": selected_count,
        "fire_rate": float(selected_count / max(labels.size, 1)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "selected_realized_delta_mean": selected_delta,
        "selected_realized_delta_sum": selected_delta_sum,
        "selected_realized_loss_sum": selected_loss,
        "selected_loss_capture_rate": float(selected_loss / total_loss) if total_loss > 0.0 else 0.0,
        "selected_veto_prevented_loss_sum": selected_loss,
        "selected_veto_forfeited_gain_sum": selected_gain,
        "selected_veto_net_gain_sum": veto_net_gain,
        "selected_veto_net_gain_mean": float(veto_net_gain / selected_count) if selected_count else 0.0,
        "selected_veto_net_gain_per_row": float(veto_net_gain / max(labels.size, 1)),
    }


def _threshold_metrics_for_selection(rows: list[dict[str, Any]], selected: np.ndarray) -> dict[str, Any]:
    labels = label_array(rows) > 0.5
    realized_delta = realized_delta_array(rows)
    realized_loss = np.maximum(0.0, -realized_delta)
    total_loss = float(np.sum(realized_loss))
    if selected.size != labels.size:
        raise ValueError("selection size must match rows")
    tp = int(np.sum(selected & labels))
    fp = int(np.sum(selected & ~labels))
    tn = int(np.sum(~selected & ~labels))
    fn = int(np.sum(~selected & labels))
    selected_count = int(np.sum(selected))
    selected_loss = float(np.sum(realized_loss[selected])) if selected_count else 0.0
    selected_gain = float(np.sum(np.maximum(0.0, realized_delta[selected]))) if selected_count else 0.0
    selected_delta_sum = float(np.sum(realized_delta[selected])) if selected_count else 0.0
    selected_delta = float(np.mean(realized_delta[selected])) if selected_count else 0.0
    veto_net_gain = -selected_delta_sum
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "rows": int(labels.size),
        "selected": selected_count,
        "fire_rate": float(selected_count / max(labels.size, 1)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "selected_realized_delta_mean": selected_delta,
        "selected_realized_delta_sum": selected_delta_sum,
        "selected_realized_loss_sum": selected_loss,
        "selected_loss_capture_rate": float(selected_loss / total_loss) if total_loss > 0.0 else 0.0,
        "selected_veto_prevented_loss_sum": selected_loss,
        "selected_veto_forfeited_gain_sum": selected_gain,
        "selected_veto_net_gain_sum": veto_net_gain,
        "selected_veto_net_gain_mean": float(veto_net_gain / selected_count) if selected_count else 0.0,
        "selected_veto_net_gain_per_row": float(veto_net_gain / max(labels.size, 1)),
    }


def threshold_metric_rows(rows: list[dict[str, Any]], thresholds: Iterable[float]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split_name, group in split_groups(rows).items():
        for threshold in thresholds:
            output.append({"split": split_name, **_threshold_metrics_for_group(group, threshold)})
    return output


def raw_score_threshold_rows(
    rows: list[dict[str, Any]],
    *,
    score_names: Iterable[str] = RAW_SCORE_NAMES,
    quantiles: Iterable[float] = DEFAULT_RAW_SCORE_QUANTILES,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    train_rows = [row for row in rows if str(row.get("split") or "") == "train"]
    threshold_source_rows = train_rows or rows
    split_map = split_groups(rows)
    for score_name in score_names:
        source_scores = raw_score_array(threshold_source_rows, score_name)
        if source_scores.size == 0:
            continue
        thresholds: list[tuple[float, float]] = []
        for quantile in quantiles:
            q = max(0.0, min(1.0, float(quantile)))
            threshold = float(np.quantile(source_scores, q))
            if math.isfinite(threshold):
                thresholds.append((q, threshold))
        deduped: list[tuple[float, float]] = []
        seen: set[float] = set()
        for quantile, threshold in thresholds:
            rounded = round(threshold, 10)
            if rounded in seen:
                continue
            seen.add(rounded)
            deduped.append((quantile, threshold))
        for split_name, group in split_map.items():
            scores = raw_score_array(group, score_name)
            for quantile, threshold in deduped:
                selected = scores >= threshold
                output.append(
                    {
                        "split": split_name,
                        "score_name": score_name,
                        "score_metric_role": raw_score_metric_role(score_name),
                        "primary_metric_source": RAW_SCORE_PRIMARY_METRIC_SOURCE,
                        "performance_claim_allowed_from_score_mean": False,
                        "threshold_source": "train_quantile" if train_rows else "all_quantile",
                        "quantile": float(quantile),
                        "threshold": float(threshold),
                        **_threshold_metrics_for_selection(group, selected),
                    }
                )
    return output


def threshold_group_metric_rows(rows: list[dict[str, Any]], thresholds: Iterable[float]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    runtime_group_fields = {"seat", "config_id", "candidate_rank_bucket"}
    group_extractors: tuple[tuple[str, Any], ...] = (
        ("seat", lambda row: str(row.get("seat") or "")),
        ("recommended_training_use", lambda row: str(row.get("recommended_training_use") or "")),
        ("local_replay_bucket", lambda row: str(row.get("local_replay_bucket") or "")),
        ("local_replay_label", lambda row: str(row.get("local_replay_label") or "")),
        ("config_id", lambda row: str(row.get("config_id") or "")),
        ("source_family", source_family),
        ("source_run", source_run),
        ("source_seed", source_seed),
        ("candidate_rank_bucket", rank_bucket),
    )
    for split_name, split_rows_for_name in split_groups(rows).items():
        for group_field, extractor in group_extractors:
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in split_rows_for_name:
                grouped[str(extractor(row))].append(row)
            for group_value in sorted(grouped):
                group = grouped[group_value]
                for threshold in thresholds:
                    output.append(
                        {
                            "split": split_name,
                            "group_field": group_field,
                            "group_value": group_value,
                            "runtime_group_available": int(group_field in runtime_group_fields),
                            **_threshold_metrics_for_group(group, threshold),
                        }
                    )
    return output


def runtime_group_value(row: dict[str, Any], group_field: str) -> str:
    if group_field == "seat":
        return str(row.get("seat") or "")
    if group_field == "config_id":
        return str(row.get("config_id") or "")
    if group_field == "candidate_rank_bucket":
        return rank_bucket(row)
    return ""


def selected_source_seed_stats(
    rows: list[dict[str, Any]], *, split: str, group_field: str, group_value: str, threshold: float
) -> dict[str, Any]:
    selected_seeds = [
        source_seed(row)
        for row in rows
        if str(row.get("split") or "") == split
        and runtime_group_value(row, group_field) == group_value
        and safe_float(row.get("risk_probability")) >= threshold
    ]
    counts = Counter(selected_seeds)
    return {
        f"{split}_selected_source_seed_count": len(counts),
        f"{split}_selected_source_seed_max_rows": max(counts.values()) if counts else 0,
    }


def selected_veto_gain_stats(
    rows: list[dict[str, Any]], *, split: str, group_field: str, group_value: str, threshold: float
) -> dict[str, Any]:
    gains = [
        -safe_float(row.get("realized_delta"))
        for row in rows
        if str(row.get("split") or "") == split
        and runtime_group_value(row, group_field) == group_value
        and safe_float(row.get("risk_probability")) >= threshold
    ]
    if not gains:
        return {
            f"{split}_selected_veto_gain_mean": 0.0,
            f"{split}_selected_veto_gain_se": 0.0,
            f"{split}_selected_veto_gain_lcb95": 0.0,
        }
    array = np.asarray(gains, dtype=np.float32)
    mean = float(np.mean(array))
    se = float(np.std(array, ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0
    return {
        f"{split}_selected_veto_gain_mean": mean,
        f"{split}_selected_veto_gain_se": se,
        f"{split}_selected_veto_gain_lcb95": float(mean - 1.96 * se),
    }


def raw_score_selected_source_seed_stats(
    rows: list[dict[str, Any]], *, split: str, score_name: str, threshold: float
) -> dict[str, Any]:
    selected_seeds = [
        source_seed(row)
        for row in rows
        if str(row.get("split") or "") == split and raw_score_value(row, score_name) >= threshold
    ]
    counts = Counter(selected_seeds)
    return {
        f"{split}_selected_source_seed_count": len(counts),
        f"{split}_selected_source_seed_max_rows": max(counts.values()) if counts else 0,
    }


def raw_score_selected_veto_gain_stats(
    rows: list[dict[str, Any]], *, split: str, score_name: str, threshold: float
) -> dict[str, Any]:
    gains = [
        -safe_float(row.get("realized_delta"))
        for row in rows
        if str(row.get("split") or "") == split and raw_score_value(row, score_name) >= threshold
    ]
    if not gains:
        return {
            f"{split}_selected_veto_gain_mean": 0.0,
            f"{split}_selected_veto_gain_se": 0.0,
            f"{split}_selected_veto_gain_lcb95": 0.0,
        }
    array = np.asarray(gains, dtype=np.float32)
    mean = float(np.mean(array))
    se = float(np.std(array, ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0
    return {
        f"{split}_selected_veto_gain_mean": mean,
        f"{split}_selected_veto_gain_se": se,
        f"{split}_selected_veto_gain_lcb95": float(mean - 1.96 * se),
    }


def runtime_group_candidate_rows(
    threshold_group_rows: list[dict[str, Any]],
    *,
    min_selected_per_split: int,
    target_selected_per_split: int,
    prediction_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    runtime_rows = [
        row
        for row in threshold_group_rows
        if int(row.get("runtime_group_available", 0)) == 1 and str(row.get("split")) in {"val", "test"}
    ]
    grouped: dict[tuple[str, str, float], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in runtime_rows:
        key = (
            str(row.get("group_field") or ""),
            str(row.get("group_value") or ""),
            float(row.get("threshold") or 0.0),
        )
        grouped[key][str(row.get("split"))] = row

    output: list[dict[str, Any]] = []
    for (group_field, group_value, threshold), split_map in grouped.items():
        val = split_map.get("val")
        test = split_map.get("test")
        if val is None or test is None:
            continue
        val_selected = int(val.get("selected", 0))
        test_selected = int(test.get("selected", 0))
        val_rows = int(val.get("rows", 0))
        test_rows = int(test.get("rows", 0))
        val_fire_rate = val_selected / max(val_rows, 1)
        test_fire_rate = test_selected / max(test_rows, 1)
        min_fire_rate = min(val_fire_rate, test_fire_rate)
        val_net = float(val.get("selected_veto_net_gain_per_row", 0.0))
        test_net = float(test.get("selected_veto_net_gain_per_row", 0.0))
        min_selected = min(val_selected, test_selected)
        min_net = min(val_net, test_net)
        stable = min_selected >= min_selected_per_split and min_net > 0.0
        near_miss = min_selected > 0 and min_net > 0.0 and not stable
        target_selected = max(1, int(target_selected_per_split))
        estimated_rows = int(math.ceil(target_selected / min_fire_rate)) if min_fire_rate > 0.0 else 0
        val_seed_stats = (
            selected_source_seed_stats(
                prediction_rows,
                split="val",
                group_field=group_field,
                group_value=group_value,
                threshold=threshold,
            )
            if prediction_rows is not None
            else {"val_selected_source_seed_count": 0, "val_selected_source_seed_max_rows": 0}
        )
        test_seed_stats = (
            selected_source_seed_stats(
                prediction_rows,
                split="test",
                group_field=group_field,
                group_value=group_value,
                threshold=threshold,
            )
            if prediction_rows is not None
            else {"test_selected_source_seed_count": 0, "test_selected_source_seed_max_rows": 0}
        )
        val_gain_stats = (
            selected_veto_gain_stats(
                prediction_rows,
                split="val",
                group_field=group_field,
                group_value=group_value,
                threshold=threshold,
            )
            if prediction_rows is not None
            else {"val_selected_veto_gain_mean": 0.0, "val_selected_veto_gain_se": 0.0, "val_selected_veto_gain_lcb95": 0.0}
        )
        test_gain_stats = (
            selected_veto_gain_stats(
                prediction_rows,
                split="test",
                group_field=group_field,
                group_value=group_value,
                threshold=threshold,
            )
            if prediction_rows is not None
            else {"test_selected_veto_gain_mean": 0.0, "test_selected_veto_gain_se": 0.0, "test_selected_veto_gain_lcb95": 0.0}
        )
        output.append(
            {
                "group_field": group_field,
                "group_value": group_value,
                "threshold": threshold,
                "min_selected_required": int(min_selected_per_split),
                "target_selected_per_split": target_selected,
                "val_rows": val_rows,
                "test_rows": test_rows,
                "val_selected": val_selected,
                "test_selected": test_selected,
                "min_selected": min_selected,
                "additional_selected_needed_for_min": max(0, int(min_selected_per_split) - min_selected),
                "additional_selected_needed_for_target": max(0, target_selected - min_selected),
                "val_fire_rate": float(val_fire_rate),
                "test_fire_rate": float(test_fire_rate),
                "min_fire_rate": float(min_fire_rate),
                "estimated_rows_per_split_for_target_selected": estimated_rows,
                **val_seed_stats,
                **test_seed_stats,
                **val_gain_stats,
                **test_gain_stats,
                "val_precision": float(val.get("precision", 0.0)),
                "test_precision": float(test.get("precision", 0.0)),
                "val_recall": float(val.get("recall", 0.0)),
                "test_recall": float(test.get("recall", 0.0)),
                "val_net_gain_per_row": val_net,
                "test_net_gain_per_row": test_net,
                "min_net_gain_per_row": min_net,
                "stable_runtime_candidate": int(stable),
                "near_miss_positive_but_underpowered": int(near_miss),
            }
        )
    output.sort(
        key=lambda row: (
            -int(row["stable_runtime_candidate"]),
            -int(row["near_miss_positive_but_underpowered"]),
            -float(row["min_net_gain_per_row"]),
            -int(row["min_selected"]),
        )
    )
    return output


def raw_score_runtime_candidate_rows(
    raw_threshold_rows: list[dict[str, Any]],
    *,
    min_selected_per_split: int,
    target_selected_per_split: int,
    prediction_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    holdout_rows = [row for row in raw_threshold_rows if str(row.get("split")) in {"val", "test"}]
    grouped: dict[tuple[str, float, float], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in holdout_rows:
        key = (
            str(row.get("score_name") or ""),
            float(row.get("quantile") or 0.0),
            float(row.get("threshold") or 0.0),
        )
        grouped[key][str(row.get("split"))] = row

    output: list[dict[str, Any]] = []
    for (score_name, quantile, threshold), split_map in grouped.items():
        val = split_map.get("val")
        test = split_map.get("test")
        if val is None or test is None:
            continue
        val_selected = int(val.get("selected", 0))
        test_selected = int(test.get("selected", 0))
        val_rows = int(val.get("rows", 0))
        test_rows = int(test.get("rows", 0))
        val_fire_rate = val_selected / max(val_rows, 1)
        test_fire_rate = test_selected / max(test_rows, 1)
        min_fire_rate = min(val_fire_rate, test_fire_rate)
        val_net = float(val.get("selected_veto_net_gain_per_row", 0.0))
        test_net = float(test.get("selected_veto_net_gain_per_row", 0.0))
        min_selected = min(val_selected, test_selected)
        min_net = min(val_net, test_net)
        stable = min_selected >= min_selected_per_split and min_net > 0.0
        near_miss = min_selected > 0 and min_net > 0.0 and not stable
        target_selected = max(1, int(target_selected_per_split))
        estimated_rows = int(math.ceil(target_selected / min_fire_rate)) if min_fire_rate > 0.0 else 0
        val_seed_stats = (
            raw_score_selected_source_seed_stats(
                prediction_rows,
                split="val",
                score_name=score_name,
                threshold=threshold,
            )
            if prediction_rows is not None
            else {"val_selected_source_seed_count": 0, "val_selected_source_seed_max_rows": 0}
        )
        test_seed_stats = (
            raw_score_selected_source_seed_stats(
                prediction_rows,
                split="test",
                score_name=score_name,
                threshold=threshold,
            )
            if prediction_rows is not None
            else {"test_selected_source_seed_count": 0, "test_selected_source_seed_max_rows": 0}
        )
        val_gain_stats = (
            raw_score_selected_veto_gain_stats(
                prediction_rows,
                split="val",
                score_name=score_name,
                threshold=threshold,
            )
            if prediction_rows is not None
            else {"val_selected_veto_gain_mean": 0.0, "val_selected_veto_gain_se": 0.0, "val_selected_veto_gain_lcb95": 0.0}
        )
        test_gain_stats = (
            raw_score_selected_veto_gain_stats(
                prediction_rows,
                split="test",
                score_name=score_name,
                threshold=threshold,
            )
            if prediction_rows is not None
            else {"test_selected_veto_gain_mean": 0.0, "test_selected_veto_gain_se": 0.0, "test_selected_veto_gain_lcb95": 0.0}
        )
        output.append(
            {
                "score_name": score_name,
                "score_metric_role": raw_score_metric_role(score_name),
                "primary_metric_source": RAW_SCORE_PRIMARY_METRIC_SOURCE,
                "performance_claim_allowed_from_score_mean": False,
                "quantile": quantile,
                "threshold": threshold,
                "threshold_source": val.get("threshold_source", ""),
                "min_selected_required": int(min_selected_per_split),
                "target_selected_per_split": target_selected,
                "val_rows": val_rows,
                "test_rows": test_rows,
                "val_selected": val_selected,
                "test_selected": test_selected,
                "min_selected": min_selected,
                "additional_selected_needed_for_min": max(0, int(min_selected_per_split) - min_selected),
                "additional_selected_needed_for_target": max(0, target_selected - min_selected),
                "val_fire_rate": float(val_fire_rate),
                "test_fire_rate": float(test_fire_rate),
                "min_fire_rate": float(min_fire_rate),
                "estimated_rows_per_split_for_target_selected": estimated_rows,
                **val_seed_stats,
                **test_seed_stats,
                **val_gain_stats,
                **test_gain_stats,
                "val_precision": float(val.get("precision", 0.0)),
                "test_precision": float(test.get("precision", 0.0)),
                "val_recall": float(val.get("recall", 0.0)),
                "test_recall": float(test.get("recall", 0.0)),
                "val_net_gain_per_row": val_net,
                "test_net_gain_per_row": test_net,
                "min_net_gain_per_row": min_net,
                "stable_runtime_candidate": int(stable),
                "near_miss_positive_but_underpowered": int(near_miss),
            }
        )
    output.sort(
        key=lambda row: (
            -int(row["stable_runtime_candidate"]),
            -int(row["near_miss_positive_but_underpowered"]),
            -float(row["min_net_gain_per_row"]),
            -int(row["min_selected"]),
        )
    )
    return output


def raw_score_baseline_rows(rows: list[dict[str, Any]], score_names: Iterable[str] = RAW_SCORE_NAMES) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split_name, group in split_groups(rows).items():
        labels = label_array(group)
        actual = labels > 0.5
        for score_name in score_names:
            scores = raw_score_array(group, score_name)
            positive_scores = scores[actual]
            negative_scores = scores[~actual]
            output.append(
                {
                    "split": split_name,
                    "score_name": score_name,
                    "score_metric_role": raw_score_metric_role(score_name),
                    "primary_metric_source": RAW_SCORE_PRIMARY_METRIC_SOURCE,
                    "performance_claim_allowed_from_score_mean": False,
                    "rows": int(labels.size),
                    "positives": int(np.sum(actual)),
                    "positive_rate": float(np.mean(labels)) if labels.size else 0.0,
                    "score_mean": float(np.mean(scores)) if scores.size else 0.0,
                    "positive_score_mean": float(np.mean(positive_scores)) if positive_scores.size else 0.0,
                    "negative_score_mean": float(np.mean(negative_scores)) if negative_scores.size else 0.0,
                    "average_precision": average_precision(labels, scores),
                    "roc_auc": roc_auc(labels, scores),
                }
            )
    return output


def decile_rows(rows: list[dict[str, Any]], *, deciles: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split_name, group in split_groups(rows).items():
        if not group:
            continue
        ordered = sorted(group, key=lambda row: safe_float(row.get("risk_probability")), reverse=True)
        n = len(ordered)
        for bucket in range(deciles):
            start = int(round(bucket * n / deciles))
            end = int(round((bucket + 1) * n / deciles))
            subset = ordered[start:end]
            if not subset:
                continue
            labels = label_array(subset)
            probabilities = probability_array(subset)
            realized_loss = realized_loss_array(subset)
            output.append(
                {
                    "split": split_name,
                    "decile": bucket + 1,
                    "score_order": "descending",
                    "rows": int(labels.size),
                    "positives": int(np.sum(labels > 0.5)),
                    "positive_rate": float(np.mean(labels)),
                    "risk_probability_min": float(np.min(probabilities)),
                    "risk_probability_max": float(np.max(probabilities)),
                    "risk_probability_mean": float(np.mean(probabilities)),
                    "realized_loss_sum": float(np.sum(realized_loss)),
                    "realized_loss_mean": float(np.mean(realized_loss)),
                }
            )
    return output


def group_breakdown_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = [subset_metrics(rows, group_field="overall", group_value="all")]
    group_defs: list[tuple[str, dict[str, list[dict[str, Any]]]]] = []
    for field in ("split", "seat", "recommended_training_use", "local_replay_bucket", "local_replay_label", "config_id"):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get(field) or "")].append(row)
        group_defs.append((field, grouped))
    source_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_run_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_seed_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rank_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source_groups[source_family(row)].append(row)
        source_run_groups[source_run(row)].append(row)
        source_seed_groups[source_seed(row)].append(row)
        rank_groups[rank_bucket(row)].append(row)
    group_defs.extend(
        [
            ("source_family", source_groups),
            ("source_run", source_run_groups),
            ("source_seed", source_seed_groups),
            ("candidate_rank_bucket", rank_groups),
        ]
    )
    for field, grouped in group_defs:
        for value in sorted(grouped):
            output.append(subset_metrics(grouped[value], group_field=field, group_value=value))
    return output


def top_error_rows(rows: list[dict[str, Any]], *, limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    false_positives = [
        row
        for row in rows
        if safe_int(row.get("label")) == 0
    ]
    false_negatives = [
        row
        for row in rows
        if safe_int(row.get("label")) == 1
    ]
    false_positives.sort(key=lambda row: safe_float(row.get("risk_probability")), reverse=True)
    false_negatives.sort(key=lambda row: safe_float(row.get("risk_probability")))
    keep_fields = (
        "row_index",
        "split",
        "source_log",
        "config_id",
        "hand_seed",
        "seat",
        "seat_swap",
        "recommended_training_use",
        "label",
        "risk_probability",
        "realized_delta",
        "realized_loss",
        "local_replay_bucket",
        "local_replay_label",
        "local_replay_delta",
        "confirm_delta",
        "confirm_delta_se",
        "predicted_delta",
        "gate_probability",
        "candidate_ev_rank",
    )

    def slim(row: dict[str, Any]) -> dict[str, Any]:
        out = {key: row.get(key, "") for key in keep_fields}
        out["source_family"] = source_family(row)
        out["source_run"] = source_run(row)
        out["source_seed"] = source_seed(row)
        out["candidate_rank_bucket"] = rank_bucket(row)
        return out

    return [slim(row) for row in false_positives[:limit]], [slim(row) for row in false_negatives[:limit]]


def audit_decision(split_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split = {str(row["split"]): row for row in split_rows}
    val_auc = safe_float(by_split.get("val", {}).get("roc_auc"))
    test_auc = safe_float(by_split.get("test", {}).get("roc_auc"))
    train_auc = safe_float(by_split.get("train", {}).get("roc_auc"))
    val_ap = safe_float(by_split.get("val", {}).get("average_precision"))
    test_ap = safe_float(by_split.get("test", {}).get("average_precision"))
    blockers: list[str] = []
    if val_auc < 0.65:
        blockers.append("val_auc_lt_0p65")
    if test_auc < 0.65:
        blockers.append("test_auc_lt_0p65")
    if train_auc - test_auc > 0.25:
        blockers.append("train_test_auc_gap_gt_0p25")
    if max(val_ap, test_ap) < 0.25:
        blockers.append("holdout_ap_lt_0p25")
    return {
        "runtime_integration_ready": not blockers,
        "runtime_integration_ready_scope": "model_quality_screen_only",
        "runtime_integration_approval": "No-Go",
        "runtime_integration_approval_reason": (
            "Requires fixed-threshold fresh seat-swap validation and explicit runtime safety review."
        ),
        "blockers": blockers,
        "train_auc": train_auc,
        "val_auc": val_auc,
        "test_auc": test_auc,
        "val_ap": val_ap,
        "test_ap": test_ap,
        "production_p2_fixed": "No-Go",
        "teacher_50k": "No-Go",
        "t1_training": "No-Go",
    }


def runtime_group_candidate_decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stable_rows = [row for row in rows if int(row.get("stable_runtime_candidate", 0)) == 1]
    near_miss_rows = [row for row in rows if int(row.get("near_miss_positive_but_underpowered", 0)) == 1]
    best_near_miss = near_miss_rows[0] if near_miss_rows else {}
    return {
        "stable_runtime_candidate_count": len(stable_rows),
        "near_miss_positive_but_underpowered_count": len(near_miss_rows),
        "runtime_group_decision": "Fresh-Heldout-Hypothesis" if near_miss_rows and not stable_rows else ("Candidate-Ready" if stable_rows else "No-Go"),
        "best_near_miss_group_field": best_near_miss.get("group_field", ""),
        "best_near_miss_group_value": best_near_miss.get("group_value", ""),
        "best_near_miss_threshold": float(best_near_miss.get("threshold", 0.0)) if best_near_miss else 0.0,
        "best_near_miss_min_selected": int(best_near_miss.get("min_selected", 0)) if best_near_miss else 0,
        "best_near_miss_min_net_gain_per_row": float(best_near_miss.get("min_net_gain_per_row", 0.0)) if best_near_miss else 0.0,
        "best_near_miss_min_lcb95": min(
            float(best_near_miss.get("val_selected_veto_gain_lcb95", 0.0)),
            float(best_near_miss.get("test_selected_veto_gain_lcb95", 0.0)),
        )
        if best_near_miss
        else 0.0,
        "best_near_miss_estimated_rows_per_split": int(best_near_miss.get("estimated_rows_per_split_for_target_selected", 0))
        if best_near_miss
        else 0,
    }


def runtime_gate_expression(row: dict[str, Any]) -> str:
    field = str(row.get("group_field") or "")
    value = str(row.get("group_value") or "")
    threshold = float(row.get("threshold", 0.0))
    if field == "candidate_rank_bucket":
        if value == "rank_1":
            guard = "candidate_ev_rank == 1"
        elif value == "rank_2_3":
            guard = "2 <= candidate_ev_rank <= 3"
        elif value == "rank_4_5":
            guard = "4 <= candidate_ev_rank <= 5"
        elif value == "rank_6_plus":
            guard = "candidate_ev_rank >= 6"
        else:
            guard = f"candidate_rank_bucket == {value!r}"
    elif field == "seat":
        guard = f"seat == {value!r}"
    elif field == "config_id":
        guard = f"config_id == {value!r}"
    else:
        guard = f"{field} == {value!r}"
    return f"risk_probability >= {threshold:.2f} and {guard}"


def fresh_heldout_plan_rows(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    plan_rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        stable = int(row.get("stable_runtime_candidate", 0))
        near_miss = int(row.get("near_miss_positive_but_underpowered", 0))
        if not stable and not near_miss:
            continue
        plan_rows.append(
            {
                "plan_status": "stable_candidate_needs_fresh_heldout" if stable else "near_miss_needs_fresh_heldout",
                "runtime_gate_expression": runtime_gate_expression(row),
                "group_field": row.get("group_field", ""),
                "group_value": row.get("group_value", ""),
                "threshold": row.get("threshold", 0.0),
                "target_selected_per_split": row.get("target_selected_per_split", 0),
                "estimated_rows_per_split": row.get("estimated_rows_per_split_for_target_selected", 0),
                "min_selected_current": row.get("min_selected", 0),
                "additional_selected_needed_for_target": row.get("additional_selected_needed_for_target", 0),
                "val_selected": row.get("val_selected", 0),
                "test_selected": row.get("test_selected", 0),
                "val_selected_source_seed_count": row.get("val_selected_source_seed_count", 0),
                "test_selected_source_seed_count": row.get("test_selected_source_seed_count", 0),
                "val_selected_veto_gain_mean": row.get("val_selected_veto_gain_mean", 0.0),
                "test_selected_veto_gain_mean": row.get("test_selected_veto_gain_mean", 0.0),
                "val_selected_veto_gain_lcb95": row.get("val_selected_veto_gain_lcb95", 0.0),
                "test_selected_veto_gain_lcb95": row.get("test_selected_veto_gain_lcb95", 0.0),
                "production_p2_fixed": "No-Go",
                "teacher_50k": "No-Go",
                "t1_training": "No-Go",
                "required_evaluation_metric": "fresh heldout realized veto utility, not training/cache LCB",
            }
        )
    return plan_rows


def write_fresh_heldout_plan(
    path: Path,
    *,
    plan_rows: list[dict[str, Any]],
    decision: dict[str, Any],
) -> None:
    lines = [
        "# HU T2 Stage8c Fresh Heldout Plan",
        "",
        "This file is a planning artifact only. It does not approve production, P2 fixed status, 50k teacher generation, or T1 training.",
        "",
        "## Current Decision",
        "",
        f"- runtime group decision: `{decision.get('runtime_group_decision', 'unknown')}`",
        f"- stable runtime candidates: `{decision.get('stable_runtime_candidate_count', 0)}`",
        f"- near-miss runtime candidates: `{decision.get('near_miss_positive_but_underpowered_count', 0)}`",
        "- production / P2 fixed: `No-Go`",
        "- 50k teacher: `No-Go`",
        "- T1 training: `No-Go`",
        "",
        "## Candidate Runtime Gates",
        "",
    ]
    if not plan_rows:
        lines.extend(
            [
                "No runtime-available group has enough positive evidence to justify a fresh heldout run.",
                "",
                "Next step: collect more diverse risk/control examples or retrain the risk head; do not run a larger runtime validation yet.",
            ]
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.extend(
        [
            "| status | runtime gate | est rows/split | current min selected | val/test selected | val/test LCB95 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in plan_rows:
        lines.append(
            "| {status} | `{gate}` | {rows} | {min_selected} | {val}/{test} | {val_lcb:.4f}/{test_lcb:.4f} |".format(
                status=row.get("plan_status", ""),
                gate=row.get("runtime_gate_expression", ""),
                rows=int(row.get("estimated_rows_per_split", 0)),
                min_selected=int(row.get("min_selected_current", 0)),
                val=int(row.get("val_selected", 0)),
                test=int(row.get("test_selected", 0)),
                val_lcb=float(row.get("val_selected_veto_gain_lcb95", 0.0)),
                test_lcb=float(row.get("test_selected_veto_gain_lcb95", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Suggested Local Command",
            "",
            "Use this as a small fresh-heldout template. Increase `-GamesPerSeed` and keep the same risk gate.",
            "",
            "```powershell",
            ".\\scripts\\Run-HuTurn2Stage8cTopkPerFireEval.ps1 `",
            "  -GamesPerSeed 1000 `",
            "  -TargetRealizedOverridesPerSeed 50 `",
            "  -Seeds \"2026063201,2026063202,2026063203\" `",
            "  -SeedStride 1000000 `",
            "  -Configs \"k5/mc64/d0/se0/confirm128/cse1/pd0/bygate_delta\" `",
            "  -Model \"models\\hu_turn2_current_fl_ev_stage8c_mixed5k_mc16.pt\" `",
            "  -Stage8cRiskModel \"models\\hu_turn2_stage8c_risk_head_opportunity_lookahead_tailmeta_1505_h16_canonical.pt\" `",
            "  -Stage8cRiskThreshold 0.35 `",
            "  -Stage8cRiskRankMin 4 `",
            "  -Stage8cRiskRankMax 5 `",
            "  -OutputDir \"outputs\\evals\\hu_turn2_stage8c_fresh_heldout_rank4_5_risk035\"",
            "```",
            "",
            "## Suggested GCP Dry Runs",
            "",
            "The GCP runner intentionally rejects ambiguous seat configs. Run first and second seat shards separately, verify `-DryRun`, then remove `-DryRun` and add `-CreateInstances` only when ready.",
            "",
            "```powershell",
            ".\\scripts\\Start-GcpHuTurn2Stage8cTopkPerFireRun.ps1 `",
            "  -DryRun `",
            "  -RunName regular-hu-t2-stage8c-fresh-heldout-rank45-risk035-first `",
            "  -GamesPerSeed 3500 `",
            "  -TargetRealizedOverridesPerSeed 50 `",
            "  -Seeds \"2026063201,2026063202,2026063203\" `",
            "  -Configs \"k5/mc64/d0/se0/confirm128/cse1/pd0/seat=first/bygate_delta\" `",
            "  -Stage8cRiskModel \"models\\hu_turn2_stage8c_risk_head_opportunity_lookahead_tailmeta_1505_h16_canonical.pt\" `",
            "  -Stage8cRiskThreshold 0.35 `",
            "  -Stage8cRiskRankMin 4 `",
            "  -Stage8cRiskRankMax 5",
            "",
            ".\\scripts\\Start-GcpHuTurn2Stage8cTopkPerFireRun.ps1 `",
            "  -DryRun `",
            "  -RunName regular-hu-t2-stage8c-fresh-heldout-rank45-risk035-second `",
            "  -GamesPerSeed 3500 `",
            "  -TargetRealizedOverridesPerSeed 50 `",
            "  -Seeds \"2026063201,2026063202,2026063203\" `",
            "  -Configs \"k5/mc64/d0/se0/confirm128/cse1/pd0/seat=second/bygate_delta\" `",
            "  -AllowRiskDataSeatScope `",
            "  -Stage8cRiskModel \"models\\hu_turn2_stage8c_risk_head_opportunity_lookahead_tailmeta_1505_h16_canonical.pt\" `",
            "  -Stage8cRiskThreshold 0.35 `",
            "  -Stage8cRiskRankMin 4 `",
            "  -Stage8cRiskRankMax 5",
            "```",
            "",
            "## Required Fresh-Heldout Rules",
            "",
            "- Use only runtime-available fields in the gate.",
            "- Measure realized veto utility on a fresh heldout split; do not reuse training/cache LCB as the performance metric.",
            "- Keep non-fired trajectories cancelable where the harness supports it, and report fired-row realized deltas separately.",
            "- Require enough selected rows before reading the sign. The current target is 50 selected rows per split.",
            "- If a near-miss remains underpowered or its LCB95 stays negative, keep T2 Stage8c No-Go.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(
    path: Path,
    manifest: dict[str, Any],
    split_rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    threshold_group_rows: list[dict[str, Any]],
    runtime_group_candidate_output: list[dict[str, Any]],
    decision: dict[str, Any],
    raw_score_rows: list[dict[str, Any]],
    raw_score_threshold_rows_output: list[dict[str, Any]],
    raw_score_candidate_output: list[dict[str, Any]],
) -> None:
    lines = [
        "# HU T2 Stage8c Risk-Head Audit",
        "",
        "This audit checks the smoke risk head only. It does not approve runtime integration, production, P2 fixed status, T1, or 50k teacher generation.",
        f"Raw score threshold sweeps are ranked by `{RAW_SCORE_PRIMARY_METRIC_SOURCE}`. Confirm-delta score means are `{RAW_CONFIRM_SCORE_ROLE}` and are not performance evidence.",
        "",
        "## Inputs",
        "",
        f"- predictions: `{manifest['predictions']}`",
        f"- rows: `{manifest['rows']}`",
        "",
        "## Split Metrics",
        "",
        "| split | rows | positives | AP | ROC AUC | Brier | prob mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in split_rows:
        lines.append(
            "| {split} | {rows} | {positives} | {ap:.4f} | {auc:.4f} | {brier:.4f} | {prob:.4f} |".format(
                split=row["split"],
                rows=int(row["rows"]),
                positives=int(row["positives"]),
                ap=float(row["average_precision"]),
                auc=float(row["roc_auc"]),
                brier=float(row["brier"]),
                prob=float(row["risk_probability_mean"]),
            )
        )
    holdout_raw: list[dict[str, Any]] = []
    for split_name in ("val", "test"):
        split_raw = [row for row in raw_score_rows if row.get("split") == split_name]
        split_raw.sort(key=lambda row: -float(row.get("roc_auc", 0.0)))
        holdout_raw.extend(split_raw[:4])
    lines.extend(
        [
            "",
            "## Raw Runtime Score Baselines",
            "",
            "| split | score | AP | ROC AUC | positive mean | negative mean |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in holdout_raw:
        lines.append(
            "| {split} | {score} | {ap:.4f} | {auc:.4f} | {pos:.4f} | {neg:.4f} |".format(
                split=row.get("split", ""),
                score=row.get("score_name", ""),
                ap=float(row.get("average_precision", 0.0)),
                auc=float(row.get("roc_auc", 0.0)),
                pos=float(row.get("positive_score_mean", 0.0)),
                neg=float(row.get("negative_score_mean", 0.0)),
            )
        )
    holdout_raw_threshold_rows = [
        row
        for row in raw_score_threshold_rows_output
        if str(row.get("split")) in {"val", "test"} and int(row.get("selected", 0)) > 0
    ]
    holdout_raw_threshold_rows.sort(key=lambda row: -float(row.get("selected_veto_net_gain_per_row", 0.0)))
    lines.extend(
        [
            "",
            "## Raw Runtime Score Veto Utility",
            "",
            "These gates use raw runtime-visible scores directly. Thresholds are train-split quantiles and are diagnostic only.",
            "",
            "| split | score | q | threshold | selected | precision | net gain/row |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in holdout_raw_threshold_rows[:20]:
        lines.append(
            "| {split} | {score} | {quantile:.2f} | {threshold:.4f} | {selected} | {precision:.3f} | {net:.4f} |".format(
                split=row.get("split", ""),
                score=row.get("score_name", ""),
                quantile=float(row.get("quantile", 0.0)),
                threshold=float(row.get("threshold", 0.0)),
                selected=int(row.get("selected", 0)),
                precision=float(row.get("precision", 0.0)),
                net=float(row.get("selected_veto_net_gain_per_row", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Raw Runtime Score Candidate Check",
            "",
            "A raw-score candidate must be positive on both val and test and meet the configured selected-row minimum before it is treated as stable.",
            "",
            "| score | q | threshold | val selected | test selected | val/test net/row | val/test LCB95 | stable | near miss |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in raw_score_candidate_output[:20]:
        lines.append(
            "| {score} | {quantile:.2f} | {threshold:.4f} | {val_selected} | {test_selected} | {val_net:.4f}/{test_net:.4f} | {val_lcb:.4f}/{test_lcb:.4f} | {stable} | {near} |".format(
                score=row.get("score_name", ""),
                quantile=float(row.get("quantile", 0.0)),
                threshold=float(row.get("threshold", 0.0)),
                val_selected=int(row.get("val_selected", 0)),
                test_selected=int(row.get("test_selected", 0)),
                val_net=float(row.get("val_net_gain_per_row", 0.0)),
                test_net=float(row.get("test_net_gain_per_row", 0.0)),
                val_lcb=float(row.get("val_selected_veto_gain_lcb95", 0.0)),
                test_lcb=float(row.get("test_selected_veto_gain_lcb95", 0.0)),
                stable=int(row.get("stable_runtime_candidate", 0)),
                near=int(row.get("near_miss_positive_but_underpowered", 0)),
            )
        )
    holdout_threshold_rows = [
        row for row in threshold_rows if str(row.get("split")) in {"val", "test"}
    ]
    holdout_threshold_rows.sort(
        key=lambda row: (str(row.get("split")), -float(row.get("selected_veto_net_gain_per_row", 0.0)))
    )
    lines.extend(
        [
            "",
            "## Veto Utility Sweep",
            "",
            "If the risk head is used as a veto, selected rows switch back to baseline. "
            "`net gain` is `prevented realized loss - forfeited realized gain`, so positive values are good.",
            "",
            "| split | threshold | selected | precision | recall | prevented loss | forfeited gain | net gain/row |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in holdout_threshold_rows:
        lines.append(
            "| {split} | {threshold:.2f} | {selected} | {precision:.3f} | {recall:.3f} | {loss:.2f} | {gain:.2f} | {net:.4f} |".format(
                split=row.get("split", ""),
                threshold=float(row.get("threshold", 0.0)),
                selected=int(row.get("selected", 0)),
                precision=float(row.get("precision", 0.0)),
                recall=float(row.get("recall", 0.0)),
                loss=float(row.get("selected_veto_prevented_loss_sum", 0.0)),
                gain=float(row.get("selected_veto_forfeited_gain_sum", 0.0)),
                net=float(row.get("selected_veto_net_gain_per_row", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Runtime-Available Group Candidate Check",
            "",
            "This table only includes group definitions that could be computed at runtime. "
            "It is still diagnostic; use it to design a fresh heldout run, not to approve production.",
            "",
            "| split | runtime? | group | value | threshold | selected | precision | net gain/row |",
            "|---|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    runtime_holdout_group_rows = [
        row
        for row in threshold_group_rows
        if str(row.get("split")) in {"val", "test"}
        and int(row.get("runtime_group_available", 0)) == 1
        and int(row.get("selected", 0)) > 0
    ]
    runtime_holdout_group_rows.sort(key=lambda row: -float(row.get("selected_veto_net_gain_per_row", 0.0)))
    for row in runtime_holdout_group_rows[:20]:
        lines.append(
            "| {split} | {runtime} | {field} | {value} | {threshold:.2f} | {selected} | {precision:.3f} | {net:.4f} |".format(
                split=row.get("split", ""),
                runtime=int(row.get("runtime_group_available", 0)),
                field=row.get("group_field", ""),
                value=row.get("group_value", ""),
                threshold=float(row.get("threshold", 0.0)),
                selected=int(row.get("selected", 0)),
                precision=float(row.get("precision", 0.0)),
                net=float(row.get("selected_veto_net_gain_per_row", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Stable Runtime Group Candidates",
            "",
            "A stable candidate must be runtime-available, positive on both val and test, "
            "and meet the configured minimum selected count in both splits.",
            "",
            "| group | value | threshold | val selected | test selected | val seeds | test seeds | min fire rate | est rows/split for target | val net/row | test net/row | val LCB95 | test LCB95 | stable | near miss |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in runtime_group_candidate_output[:20]:
        lines.append(
            "| {field} | {value} | {threshold:.2f} | {val_selected} | {test_selected} | {val_seeds} | {test_seeds} | {fire_rate:.4f} | {est_rows} | {val_net:.4f} | {test_net:.4f} | {val_lcb:.4f} | {test_lcb:.4f} | {stable} | {near} |".format(
                field=row.get("group_field", ""),
                value=row.get("group_value", ""),
                threshold=float(row.get("threshold", 0.0)),
                val_selected=int(row.get("val_selected", 0)),
                test_selected=int(row.get("test_selected", 0)),
                val_seeds=int(row.get("val_selected_source_seed_count", 0)),
                test_seeds=int(row.get("test_selected_source_seed_count", 0)),
                fire_rate=float(row.get("min_fire_rate", 0.0)),
                est_rows=int(row.get("estimated_rows_per_split_for_target_selected", 0)),
                val_net=float(row.get("val_net_gain_per_row", 0.0)),
                test_net=float(row.get("test_net_gain_per_row", 0.0)),
                val_lcb=float(row.get("val_selected_veto_gain_lcb95", 0.0)),
                test_lcb=float(row.get("test_selected_veto_gain_lcb95", 0.0)),
                stable=int(row.get("stable_runtime_candidate", 0)),
                near=int(row.get("near_miss_positive_but_underpowered", 0)),
            )
        )
    diagnostic_holdout_group_rows = [
        row
        for row in threshold_group_rows
        if str(row.get("split")) in {"val", "test"}
        and int(row.get("runtime_group_available", 0)) == 0
        and int(row.get("selected", 0)) > 0
    ]
    diagnostic_holdout_group_rows.sort(key=lambda row: -float(row.get("selected_veto_net_gain_per_row", 0.0)))
    lines.extend(
        [
            "",
            "## Top Diagnostic-Only Group Veto Utility",
            "",
            "These rows may use labels, source names, or post-hoc fields. They are useful for diagnosis only and are not deployable runtime gates.",
            "",
            "| split | group | value | threshold | selected | precision | net gain/row |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in diagnostic_holdout_group_rows[:20]:
        lines.append(
            "| {split} | {field} | {value} | {threshold:.2f} | {selected} | {precision:.3f} | {net:.4f} |".format(
                split=row.get("split", ""),
                field=row.get("group_field", ""),
                value=row.get("group_value", ""),
                threshold=float(row.get("threshold", 0.0)),
                selected=int(row.get("selected", 0)),
                precision=float(row.get("precision", 0.0)),
                net=float(row.get("selected_veto_net_gain_per_row", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- target mode: `{manifest.get('target_mode', 'unknown')}`",
            f"- risk probability semantics: `{manifest.get('risk_probability_semantics', '')}`",
            f"- model-quality screen ready: `{'true' if decision['runtime_integration_ready'] else 'false'}`",
            f"- runtime integration approval: `{manifest.get('runtime_integration_approval', 'No-Go')}`",
            f"- blockers: `{', '.join(decision['blockers']) if decision['blockers'] else 'none'}`",
            f"- runtime group decision: `{manifest.get('runtime_group_candidate_decision', {}).get('runtime_group_decision', 'unknown')}`",
            f"- stable runtime group candidates: `{manifest.get('runtime_group_candidate_decision', {}).get('stable_runtime_candidate_count', 0)}`",
            f"- near-miss runtime group candidates: `{manifest.get('runtime_group_candidate_decision', {}).get('near_miss_positive_but_underpowered_count', 0)}`",
            f"- stable raw-score candidates: `{manifest.get('raw_score_candidate_decision', {}).get('stable_runtime_candidate_count', 0)}`",
            f"- near-miss raw-score candidates: `{manifest.get('raw_score_candidate_decision', {}).get('near_miss_positive_but_underpowered_count', 0)}`",
            "- production / P2 fixed: `No-Go`",
            "- 50k teacher: `No-Go`",
            "- T1 training: `No-Go`",
            "",
            "## Interpretation",
            "",
        ]
    )
    if decision["runtime_integration_ready"]:
        lines.append(
            "- The head has enough holdout signal for a further fixed-threshold fresh runtime hypothesis test, not production or direct runtime integration."
        )
    else:
        lines.extend(
            [
                "- The smoke risk head is execution-pass but decision No-Go.",
                "- Holdout AP/AUC are too weak and/or the train-test gap is too large.",
                "- Do not wire this model into runtime gates. Use the audit outputs to guide more diverse risk/control data collection or a lower-capacity/regularized retrain.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(
    rows: list[dict[str, Any]],
    *,
    thresholds: list[float],
    deciles: int,
    top_errors: int,
    output_dir: Path,
    predictions_path: Path,
    runtime_group_min_selected: int = 10,
    runtime_group_target_selected: int = 50,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_rows = split_metric_rows(rows)
    threshold_rows = threshold_metric_rows(rows, thresholds)
    threshold_group_rows = threshold_group_metric_rows(rows, thresholds)
    runtime_group_candidate_output = runtime_group_candidate_rows(
        threshold_group_rows,
        min_selected_per_split=max(1, runtime_group_min_selected),
        target_selected_per_split=max(1, runtime_group_target_selected),
        prediction_rows=rows,
    )
    fresh_heldout_plan_output = fresh_heldout_plan_rows(runtime_group_candidate_output)
    decile_output = decile_rows(rows, deciles=deciles)
    group_output = group_breakdown_rows(rows)
    raw_score_output = raw_score_baseline_rows(rows)
    raw_score_threshold_output = raw_score_threshold_rows(rows)
    raw_score_candidate_output = raw_score_runtime_candidate_rows(
        raw_score_threshold_output,
        min_selected_per_split=max(1, runtime_group_min_selected),
        target_selected_per_split=max(1, runtime_group_target_selected),
        prediction_rows=rows,
    )
    false_positive_rows, false_negative_rows = top_error_rows(rows, limit=top_errors)
    decision = audit_decision(split_rows)
    target_mode = infer_target_mode(rows)
    runtime_group_decision = runtime_group_candidate_decision(runtime_group_candidate_output)
    raw_score_candidate_decision = runtime_group_candidate_decision(raw_score_candidate_output)
    manifest = {
        "schema": "hu_turn2_stage8c_risk_head_audit_v1",
        "predictions": str(predictions_path),
        "output_dir": str(output_dir),
        "rows": len(rows),
        "target_mode": target_mode,
        "risk_probability_semantics": (
            "probability of local EV negative label"
            if target_mode == "local_ev_negative"
            else "probability of whole-game risk label"
            if target_mode == "whole_game_risk"
            else "probability of configured audit label"
        ),
        "runtime_integration_approval": "No-Go",
        "raw_score_primary_metric_source": RAW_SCORE_PRIMARY_METRIC_SOURCE,
        "raw_confirm_score_role": RAW_CONFIRM_SCORE_ROLE,
        "raw_confirm_score_performance_claim_allowed": RAW_CONFIRM_SCORE_PERFORMANCE_CLAIM_ALLOWED,
        "runtime_integration_approval_reason": (
            "Audit metrics are screening evidence only; runtime use still requires fresh fixed-threshold seat-swap validation."
        ),
        "thresholds": thresholds,
        "deciles": deciles,
        "top_errors": top_errors,
        "runtime_group_min_selected": max(1, runtime_group_min_selected),
        "runtime_group_target_selected": max(1, runtime_group_target_selected),
        "decision": decision,
        "runtime_group_candidate_decision": runtime_group_decision,
        "raw_score_candidate_decision": raw_score_candidate_decision,
        "artifacts": [
            "risk_head_audit_summary.md",
            "risk_head_split_metrics.csv",
            "risk_head_threshold_sweep.csv",
            "risk_head_threshold_group_sweep.csv",
            "risk_head_runtime_group_candidates.csv",
            "risk_head_fresh_heldout_plan.csv",
            "risk_head_fresh_heldout_plan.md",
            "risk_head_deciles.csv",
            "risk_head_group_breakdown.csv",
            "risk_head_raw_score_baselines.csv",
            "risk_head_raw_score_threshold_sweep.csv",
            "risk_head_raw_score_runtime_candidates.csv",
            "risk_head_top_false_positive_predictions.jsonl",
            "risk_head_top_false_negative_predictions.jsonl",
        ],
    }
    write_csv(output_dir / "risk_head_split_metrics.csv", split_rows)
    write_csv(output_dir / "risk_head_threshold_sweep.csv", threshold_rows)
    write_csv(output_dir / "risk_head_threshold_group_sweep.csv", threshold_group_rows)
    write_csv(output_dir / "risk_head_runtime_group_candidates.csv", runtime_group_candidate_output)
    write_csv(output_dir / "risk_head_fresh_heldout_plan.csv", fresh_heldout_plan_output)
    write_csv(output_dir / "risk_head_deciles.csv", decile_output)
    write_csv(output_dir / "risk_head_group_breakdown.csv", group_output)
    write_csv(output_dir / "risk_head_raw_score_baselines.csv", raw_score_output)
    write_csv(output_dir / "risk_head_raw_score_threshold_sweep.csv", raw_score_threshold_output)
    write_csv(output_dir / "risk_head_raw_score_runtime_candidates.csv", raw_score_candidate_output)
    write_jsonl(output_dir / "risk_head_top_false_positive_predictions.jsonl", false_positive_rows)
    write_jsonl(output_dir / "risk_head_top_false_negative_predictions.jsonl", false_negative_rows)
    (output_dir / "risk_head_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary(
        output_dir / "risk_head_audit_summary.md",
        manifest,
        split_rows,
        threshold_rows,
        threshold_group_rows,
        runtime_group_candidate_output,
        decision,
        raw_score_output,
        raw_score_threshold_output,
        raw_score_candidate_output,
    )
    write_fresh_heldout_plan(
        output_dir / "risk_head_fresh_heldout_plan.md",
        plan_rows=fresh_heldout_plan_output,
        decision=runtime_group_decision,
    )
    return manifest


def main() -> None:
    args = parse_args()
    rows = load_prediction_rows(args.predictions)
    manifest = run_audit(
        rows,
        thresholds=parse_thresholds(args.thresholds),
        deciles=max(1, args.deciles),
        top_errors=max(0, args.top_errors),
        output_dir=args.output_dir,
        predictions_path=args.predictions,
        runtime_group_min_selected=max(1, args.runtime_group_min_selected),
        runtime_group_target_selected=max(1, args.runtime_group_target_selected),
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
