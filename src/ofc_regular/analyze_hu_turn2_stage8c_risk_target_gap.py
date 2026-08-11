"""Analyze why Stage8c whole-game risk labels are hard to learn.

This is a diagnostic step only. It compares local T2 replay evidence with
whole-game realized labels and reports whether the available rows contain the
downstream trajectory fields needed to learn whole-game risk.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_TARGET_NAME = "topk_counterfactual_loss_targets_merged.jsonl"
FALLBACK_TARGET_NAME = "topk_counterfactual_loss_targets.jsonl"
DEFAULT_OUTPUT_DIR = Path("outputs/training/hu_turn2_stage8c_risk_target_gap_analysis")
PRIMARY_METRIC_SOURCE = "realized_whole_game_and_local_replay_delta"
CONFIRM_DELTA_METRIC_ROLE = "runtime_feature_diagnostic_only"
CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED = False
DOWNSTREAM_FIELDS = (
    "post_t2_candidate_board",
    "post_t2_baseline_board",
    "t3_decision_summary",
    "final_board_hero",
    "final_board_opponent",
    "hero_foul",
    "opponent_foul",
    "hero_fl_entry",
    "hero_fl_stay",
    "opponent_fl_entry",
    "hero_royalty",
    "opponent_royalty",
    "line_score_delta",
    "scoop_delta",
    "downstream_override_fired",
    "paired_future_delta_summary",
)
TRAJECTORY_COMPONENT_FIELDS = (
    "terminal_score_vs_baseline",
    "royalty_delta_vs_baseline",
    "fl_delta_vs_baseline",
    "line_score_delta_vs_baseline",
    "scoop_delta_vs_baseline",
    "foul_delta_vs_baseline",
    "hero_royalty_vs_baseline",
    "opponent_royalty_vs_baseline",
    "hero_fl_value_vs_baseline",
    "opponent_fl_value_vs_baseline",
    "candidate_terminal_score",
    "baseline_terminal_score",
    "candidate_royalty_delta",
    "baseline_royalty_delta",
    "candidate_fl_delta",
    "baseline_fl_delta",
    "candidate_line_score_delta",
    "baseline_line_score_delta",
    "candidate_scoop_delta",
    "baseline_scoop_delta",
    "candidate_foul_delta",
    "baseline_foul_delta",
)
TRAJECTORY_BOOLEAN_FIELDS = (
    "candidate_hero_foul",
    "baseline_hero_foul",
    "candidate_opponent_foul",
    "baseline_opponent_foul",
    "candidate_hero_fl_entry",
    "baseline_hero_fl_entry",
    "candidate_hero_fl_stay",
    "baseline_hero_fl_stay",
    "candidate_opponent_fl_entry",
    "baseline_opponent_fl_entry",
    "candidate_opponent_fl_stay",
    "baseline_opponent_fl_stay",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection-dir",
        type=Path,
        action="append",
        default=[],
        help=f"Directory containing {DEFAULT_TARGET_NAME}. Can be repeated.",
    )
    parser.add_argument("--input-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=50)
    return parser.parse_args()


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


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalize_path(value: Any) -> str:
    return str(value or "").replace("/", "\\").lower()


def row_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        normalize_path(row.get("source_log", "")),
        str(row.get("config_id", "")),
        str(row.get("hand_seed", "")),
        str(row.get("seat", "")),
        str(safe_int(row.get("candidate_index"), -1)),
        str(safe_int(row.get("baseline_index"), -1)),
    )


def target_paths(collection_dirs: Iterable[Path], input_jsonls: Iterable[Path]) -> list[Path]:
    paths = []
    for directory in collection_dirs:
        merged = directory / DEFAULT_TARGET_NAME
        fallback = directory / FALLBACK_TARGET_NAME
        paths.append(merged if merged.exists() else fallback)
    paths.extend(input_jsonls)
    if not paths:
        raise SystemExit("at least one --collection-dir or --input-jsonl is required")
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing[0])
    return paths


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                row["_gap_source_path"] = str(path)
                rows.append(row)
    return rows


def load_rows(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for path in paths:
        rows = read_jsonl(path)
        sources.append({"source_path": str(path), "rows": len(rows)})
        for row in rows:
            key = row_key(row)
            current = by_key.get(key)
            if current is None:
                by_key[key] = row
                continue
            if safe_int(row.get("local_replay_future_samples")) > safe_int(
                current.get("local_replay_future_samples")
            ):
                by_key[key] = row
    return sorted(by_key.values(), key=row_key), sources


def recommended_use(row: dict[str, Any]) -> str:
    use = str(row.get("recommended_training_use", "")).strip()
    if use:
        return use
    if truthy(row.get("use_for_local_ev_hard_negative")):
        return "local_ev_hard_negative"
    if truthy(row.get("use_for_whole_game_risk_head")) and safe_float(row.get("realized_delta")) >= 0.0:
        return "whole_game_non_loss_control"
    if truthy(row.get("use_for_whole_game_risk_head")):
        return "whole_game_risk_only"
    if truthy(row.get("requires_local_replay")):
        return "requires_local_replay"
    return "other"


def source_family(row: dict[str, Any]) -> str:
    value = " ".join(
        str(row.get(field) or "")
        for field in (
            "collection_source_path",
            "target_source_path",
            "source_log",
            "_gap_source_path",
        )
    ).lower()
    if "c4_allfired_mc512_replay" in value or "topk_all_fired_deduped" in value:
        return "c4_allfired_mc512_replay"
    if "30seed_plus_local_risk_veto_rank2" in value:
        return "stage8c_30seed_local_risk_veto"
    if "fresh_schema_target50" in value:
        return "stage8c_fresh_schema"
    if "risk-expand" in value:
        return "risk_expand"
    if "risk-fill" in value:
        return "risk_fill"
    if "risk-second" in value:
        return "risk_second"
    if "risk_control_extraction_smoke" in value or "risk-control" in value:
        return "risk_control_smoke"
    if not value:
        return "unknown"
    return "other"


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


def present(value: Any) -> bool:
    return value not in (None, "", [], {})


def downstream_complete(row: dict[str, Any]) -> bool:
    return all(present(row.get(field)) for field in DOWNSTREAM_FIELDS)


def confirm_z(row: dict[str, Any]) -> float:
    se = max(safe_float(row.get("confirm_delta_se")), 1e-9)
    return safe_float(row.get("confirm_delta")) / se


def normalized_row(row: dict[str, Any]) -> dict[str, Any]:
    realized_delta = safe_float(row.get("realized_delta"))
    local_delta = safe_float(row.get("local_replay_delta"))
    out = {
        "source_log": row.get("source_log", ""),
        "source_family": source_family(row),
        "config_id": row.get("config_id", ""),
        "hand_seed": row.get("hand_seed", ""),
        "hand_id": row.get("hand_id", ""),
        "seat": row.get("seat", ""),
        "seat_swap": row.get("seat_swap", ""),
        "recommended_training_use": recommended_use(row),
        "realized_delta": realized_delta,
        "realized_loss": max(0.0, -realized_delta),
        "local_replay_bucket": row.get("local_replay_bucket", ""),
        "local_replay_label": row.get("local_replay_label", ""),
        "local_replay_delta": local_delta,
        "local_replay_delta_se": safe_float(row.get("local_replay_delta_se")),
        "local_replay_lcb196": safe_float(row.get("local_replay_lcb196")),
        "confirm_delta": safe_float(row.get("confirm_delta")),
        "confirm_delta_se": safe_float(row.get("confirm_delta_se")),
        "confirm_delta_z": confirm_z(row),
        "predicted_delta": safe_float(row.get("predicted_delta")),
        "gate_probability": safe_float(row.get("gate_probability")),
        "candidate_ev_rank": safe_int(row.get("candidate_ev_rank"), 9999),
        "candidate_rank_bucket": rank_bucket(row),
        "candidate_index": safe_int(row.get("candidate_index"), -1),
        "baseline_index": safe_int(row.get("baseline_index"), -1),
        "local_positive": int(str(row.get("local_replay_label", "")).lower() == "positive"),
        "local_negative": int(str(row.get("local_replay_label", "")).lower() == "negative"),
        "local_gray": int(str(row.get("local_replay_label", "")).lower() == "gray"),
        "collection_source_path": row.get("collection_source_path", ""),
        "_gap_source_path": row.get("_gap_source_path", ""),
    }
    for field in DOWNSTREAM_FIELDS:
        out[f"has_{field}"] = int(present(row.get(field)))
    out["downstream_fields_present"] = sum(out[f"has_{field}"] for field in DOWNSTREAM_FIELDS)
    out["downstream_fields_total"] = len(DOWNSTREAM_FIELDS)
    out["downstream_trajectory_complete"] = int(out["downstream_fields_present"] == len(DOWNSTREAM_FIELDS))
    for field in TRAJECTORY_COMPONENT_FIELDS:
        out[field] = safe_float(row.get(field))
    for field in TRAJECTORY_BOOLEAN_FIELDS:
        out[field] = int(truthy(row.get(field)))
    out["hero_foul_changed"] = int(out["candidate_hero_foul"] != out["baseline_hero_foul"])
    out["opponent_foul_changed"] = int(out["candidate_opponent_foul"] != out["baseline_opponent_foul"])
    out["hero_fl_entry_changed"] = int(out["candidate_hero_fl_entry"] != out["baseline_hero_fl_entry"])
    out["hero_fl_stay_changed"] = int(out["candidate_hero_fl_stay"] != out["baseline_hero_fl_stay"])
    out["opponent_fl_entry_changed"] = int(
        out["candidate_opponent_fl_entry"] != out["baseline_opponent_fl_entry"]
    )
    out["opponent_fl_stay_changed"] = int(
        out["candidate_opponent_fl_stay"] != out["baseline_opponent_fl_stay"]
    )
    return out


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def percentile(values: Iterable[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, math.ceil((q / 100.0) * len(values)) - 1))
    return values[index]


def roc_auc_from_scores(labels: list[int], scores: list[float]) -> float:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return 0.0
    combined = [(score, 1) for score in positives] + [(score, 0) for score in negatives]
    combined.sort(key=lambda item: item[0])
    ranks = [0.0] * len(combined)
    start = 0
    while start < len(combined):
        end = start + 1
        while end < len(combined) and combined[end][0] == combined[start][0]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        for index in range(start, end):
            ranks[index] = avg_rank
        start = end
    pos_rank_sum = sum(rank for rank, item in zip(ranks, combined) if item[1] == 1)
    return float((pos_rank_sum - len(positives) * (len(positives) + 1) / 2.0) / (len(positives) * len(negatives)))


def _component_summary(field: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [safe_float(row.get(field)) for row in rows]
    return {
        "mean": mean(values),
        "p05": percentile(values, 5),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "negative_rows": sum(value < 0.0 for value in values),
        "positive_rows": sum(value > 0.0 for value in values),
        "zero_rows": sum(value == 0.0 for value in values),
    }


def trajectory_component_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    risk = [row for row in rows if row.get("recommended_training_use") == "whole_game_risk_only"]
    controls = [row for row in rows if row.get("recommended_training_use") == "whole_game_non_loss_control"]
    local_hard = [row for row in rows if row.get("recommended_training_use") == "local_ev_hard_negative"]
    output: list[dict[str, Any]] = []
    for field in (*TRAJECTORY_COMPONENT_FIELDS, *TRAJECTORY_BOOLEAN_FIELDS):
        risk_summary = _component_summary(field, risk)
        control_summary = _component_summary(field, controls)
        hard_summary = _component_summary(field, local_hard)
        output.append(
            {
                "field": field,
                "risk_rows": len(risk),
                "control_rows": len(controls),
                "local_hard_rows": len(local_hard),
                "risk_mean": risk_summary["mean"],
                "control_mean": control_summary["mean"],
                "local_hard_mean": hard_summary["mean"],
                "risk_minus_control_mean": risk_summary["mean"] - control_summary["mean"],
                "risk_p05": risk_summary["p05"],
                "risk_p50": risk_summary["p50"],
                "risk_p95": risk_summary["p95"],
                "control_p05": control_summary["p05"],
                "control_p50": control_summary["p50"],
                "control_p95": control_summary["p95"],
                "risk_negative_rows": risk_summary["negative_rows"],
                "risk_negative_rate": risk_summary["negative_rows"] / len(risk) if risk else 0.0,
                "control_negative_rows": control_summary["negative_rows"],
                "control_negative_rate": control_summary["negative_rows"] / len(controls) if controls else 0.0,
                "risk_positive_rows": risk_summary["positive_rows"],
                "risk_positive_rate": risk_summary["positive_rows"] / len(risk) if risk else 0.0,
                "control_positive_rows": control_summary["positive_rows"],
                "control_positive_rate": control_summary["positive_rows"] / len(controls) if controls else 0.0,
            }
        )
    for field in (
        "hero_foul_changed",
        "opponent_foul_changed",
        "hero_fl_entry_changed",
        "hero_fl_stay_changed",
        "opponent_fl_entry_changed",
        "opponent_fl_stay_changed",
    ):
        risk_summary = _component_summary(field, risk)
        control_summary = _component_summary(field, controls)
        output.append(
            {
                "field": field,
                "risk_rows": len(risk),
                "control_rows": len(controls),
                "local_hard_rows": len(local_hard),
                "risk_mean": risk_summary["mean"],
                "control_mean": control_summary["mean"],
                "local_hard_mean": _component_summary(field, local_hard)["mean"],
                "risk_minus_control_mean": risk_summary["mean"] - control_summary["mean"],
                "risk_p05": risk_summary["p05"],
                "risk_p50": risk_summary["p50"],
                "risk_p95": risk_summary["p95"],
                "control_p05": control_summary["p05"],
                "control_p50": control_summary["p50"],
                "control_p95": control_summary["p95"],
                "risk_negative_rows": 0,
                "risk_negative_rate": 0.0,
                "control_negative_rows": 0,
                "control_negative_rate": 0.0,
                "risk_positive_rows": risk_summary["positive_rows"],
                "risk_positive_rate": risk_summary["positive_rows"] / len(risk) if risk else 0.0,
                "control_positive_rows": control_summary["positive_rows"],
                "control_positive_rate": control_summary["positive_rows"] / len(controls) if controls else 0.0,
            }
        )
    return output


def trajectory_feature_auc_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    binary_rows = [
        row
        for row in rows
        if row.get("recommended_training_use") in {"whole_game_risk_only", "whole_game_non_loss_control"}
    ]
    labels = [1 if row.get("recommended_training_use") == "whole_game_risk_only" else 0 for row in binary_rows]
    fields = (
        *TRAJECTORY_COMPONENT_FIELDS,
        *TRAJECTORY_BOOLEAN_FIELDS,
        "hero_foul_changed",
        "opponent_foul_changed",
        "hero_fl_entry_changed",
        "hero_fl_stay_changed",
        "opponent_fl_entry_changed",
        "opponent_fl_stay_changed",
        "predicted_delta",
        "confirm_delta",
        "confirm_delta_z",
        "local_replay_delta",
    )
    output = []
    for field in fields:
        values = [safe_float(row.get(field)) for row in binary_rows]
        auc = roc_auc_from_scores(labels, values)
        output.append(
            {
                "field": field,
                "feature_metric_role": (
                    CONFIRM_DELTA_METRIC_ROLE if field.startswith("confirm_delta") else "diagnostic_feature"
                ),
                "rows": len(binary_rows),
                "positives": sum(labels),
                "auc_high_means_risk": auc,
                "auc_low_means_risk": 1.0 - auc,
                "best_direction": "high" if auc >= 0.5 else "low",
                "best_auc": max(auc, 1.0 - auc),
                "risk_mean": mean(value for label, value in zip(labels, values) if label == 1),
                "control_mean": mean(value for label, value in zip(labels, values) if label == 0),
            }
        )
    output.sort(key=lambda row: safe_float(row.get("best_auc")), reverse=True)
    return output


def summarize_subset(group_field: str, group_value: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    realized = [safe_float(row.get("realized_delta")) for row in rows]
    losses = [safe_float(row.get("realized_loss")) for row in rows]
    local = [safe_float(row.get("local_replay_delta")) for row in rows]
    lcb = [safe_float(row.get("local_replay_lcb196")) for row in rows]
    return {
        "group_field": group_field,
        "group_value": group_value,
        "rows": len(rows),
        "whole_game_risk_only": sum(row["recommended_training_use"] == "whole_game_risk_only" for row in rows),
        "whole_game_non_loss_control": sum(
            row["recommended_training_use"] == "whole_game_non_loss_control" for row in rows
        ),
        "local_ev_hard_negative": sum(row["recommended_training_use"] == "local_ev_hard_negative" for row in rows),
        "local_positive_rows": sum(safe_int(row.get("local_positive")) for row in rows),
        "local_negative_rows": sum(safe_int(row.get("local_negative")) for row in rows),
        "local_gray_rows": sum(safe_int(row.get("local_gray")) for row in rows),
        "realized_delta_mean": mean(realized),
        "realized_loss_mean": mean(losses),
        "realized_loss_p90": percentile(losses, 90),
        "realized_loss_p95": percentile(losses, 95),
        "realized_loss_max": max(losses, default=0.0),
        "local_replay_delta_mean": mean(local),
        "local_replay_lcb196_mean": mean(lcb),
        "confirm_delta_mean": mean(safe_float(row.get("confirm_delta")) for row in rows),
        "confirm_delta_z_mean": mean(safe_float(row.get("confirm_delta_z")) for row in rows),
        "confirm_delta_metric_role": CONFIRM_DELTA_METRIC_ROLE,
        "confirm_delta_performance_claim_allowed": CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED,
        "primary_metric_source": PRIMARY_METRIC_SOURCE,
        "predicted_delta_mean": mean(safe_float(row.get("predicted_delta")) for row in rows),
        "gate_probability_mean": mean(safe_float(row.get("gate_probability")) for row in rows),
    }


def breakdown_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = [summarize_subset("overall", "all", rows)]
    for field in (
        "recommended_training_use",
        "local_replay_bucket",
        "local_replay_label",
        "seat",
        "source_family",
        "candidate_rank_bucket",
        "config_id",
    ):
        for value in sorted({str(row.get(field, "")) for row in rows}):
            output.append(
                summarize_subset(field, value, [row for row in rows if str(row.get(field, "")) == value])
            )
    return output


def matrix_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("local_replay_bucket", "")),
                str(row.get("local_replay_label", "")),
                str(row.get("recommended_training_use", "")),
            )
        ].append(row)
    output = []
    for (bucket, label, use), subset in sorted(grouped.items()):
        summary = summarize_subset("matrix", f"{bucket}/{label}/{use}", subset)
        summary.update(
            {
                "local_replay_bucket": bucket,
                "local_replay_label": label,
                "recommended_training_use": use,
            }
        )
        output.append(summary)
    return output


def field_coverage_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    total = len(raw_rows)
    for field in DOWNSTREAM_FIELDS:
        present_count = sum(1 for row in raw_rows if present(row.get(field)))
        output.append(
            {
                "field": field,
                "present_rows": present_count,
                "missing_rows": total - present_count,
                "present_rate": present_count / total if total else 0.0,
            }
        )
    return output


def _coverage_group_value(row: dict[str, Any], group_field: str) -> str:
    if group_field == "source_family":
        return source_family(row)
    if group_field == "recommended_training_use":
        return recommended_use(row)
    if group_field == "collection_source_path":
        return str(row.get("collection_source_path") or row.get("_gap_source_path") or "")
    return str(row.get(group_field, ""))


def field_coverage_by_group_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    group_fields = ("source_family", "realized_delta_basis", "recommended_training_use")
    for group_field in group_fields:
        values = sorted({_coverage_group_value(row, group_field) for row in raw_rows})
        for value in values:
            subset = [row for row in raw_rows if _coverage_group_value(row, group_field) == value]
            total = len(subset)
            complete_count = sum(1 for row in subset if downstream_complete(row))
            output.append(
                {
                    "group_field": group_field,
                    "group_value": value,
                    "field": "all_downstream_fields",
                    "rows": total,
                    "present_rows": complete_count,
                    "missing_rows": total - complete_count,
                    "present_rate": complete_count / total if total else 0.0,
                }
            )
            for field in DOWNSTREAM_FIELDS:
                present_count = sum(1 for row in subset if present(row.get(field)))
                output.append(
                    {
                        "group_field": group_field,
                        "group_value": value,
                        "field": field,
                        "rows": total,
                        "present_rows": present_count,
                        "missing_rows": total - present_count,
                        "present_rate": present_count / total if total else 0.0,
                    }
                )
    return output


def metric_rows(rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]], sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(str(row.get("recommended_training_use")) for row in rows)
    risk_only = [row for row in rows if row.get("recommended_training_use") == "whole_game_risk_only"]
    controls = [row for row in rows if row.get("recommended_training_use") == "whole_game_non_loss_control"]
    local_hard = [row for row in rows if row.get("recommended_training_use") == "local_ev_hard_negative"]
    risk_local_positive = [row for row in risk_only if row.get("local_replay_label") == "positive"]
    risk_local_gray = [row for row in risk_only if row.get("local_replay_label") == "gray"]
    risk_local_negative = [row for row in risk_only if row.get("local_replay_label") == "negative"]
    controls_local_negative = [row for row in controls if row.get("local_replay_label") == "negative"]
    coverage = field_coverage_rows(raw_rows)
    missing_downstream_fields = sum(1 for row in coverage if row["present_rows"] == 0)
    partial_downstream_fields = sum(1 for row in coverage if 0 < safe_int(row.get("present_rows")) < len(raw_rows))
    downstream_complete_rows = sum(1 for row in raw_rows if downstream_complete(row))
    metrics: list[dict[str, Any]] = [
        {"metric": "input_paths", "value": len(sources)},
        {"metric": "input_rows", "value": sum(safe_int(row.get("rows")) for row in sources)},
        {"metric": "deduped_rows", "value": len(rows)},
        {"metric": "whole_game_risk_only", "value": counts["whole_game_risk_only"]},
        {"metric": "whole_game_non_loss_control", "value": counts["whole_game_non_loss_control"]},
        {"metric": "local_ev_hard_negative", "value": counts["local_ev_hard_negative"]},
        {"metric": "risk_only_local_positive", "value": len(risk_local_positive)},
        {"metric": "risk_only_local_positive_share", "value": len(risk_local_positive) / len(risk_only) if risk_only else 0.0},
        {"metric": "risk_only_local_gray", "value": len(risk_local_gray)},
        {"metric": "risk_only_local_negative", "value": len(risk_local_negative)},
        {"metric": "non_loss_controls_local_negative", "value": len(controls_local_negative)},
        {
            "metric": "non_loss_controls_local_negative_share",
            "value": len(controls_local_negative) / len(controls) if controls else 0.0,
        },
        {"metric": "local_hard_negative_rows", "value": len(local_hard)},
        {
            "metric": "risk_only_mean_local_delta",
            "value": mean(safe_float(row.get("local_replay_delta")) for row in risk_only),
        },
        {
            "metric": "risk_only_mean_realized_loss",
            "value": mean(safe_float(row.get("realized_loss")) for row in risk_only),
        },
        {
            "metric": "controls_mean_local_delta",
            "value": mean(safe_float(row.get("local_replay_delta")) for row in controls),
        },
        {"metric": "downstream_fields_total", "value": len(DOWNSTREAM_FIELDS)},
        {"metric": "downstream_fields_missing_everywhere", "value": missing_downstream_fields},
        {"metric": "downstream_fields_partial_missing", "value": partial_downstream_fields},
        {"metric": "downstream_trajectory_complete_rows", "value": downstream_complete_rows},
        {
            "metric": "downstream_trajectory_complete_share",
            "value": downstream_complete_rows / len(raw_rows) if raw_rows else 0.0,
        },
    ]
    return metrics


def diagnostic_flags(metrics: list[dict[str, Any]]) -> list[str]:
    values = {str(row["metric"]): row["value"] for row in metrics}
    flags: list[str] = []
    if safe_float(values.get("risk_only_local_positive_share")) >= 0.50:
        flags.append("whole_game_risk_is_mostly_locally_positive")
    if safe_int(values.get("risk_only_local_negative")) == 0:
        flags.append("risk_only_has_no_local_negative_rows")
    if safe_int(values.get("non_loss_controls_local_negative")) > 0:
        flags.append("local_negative_can_still_be_whole_game_non_loss")
    if safe_int(values.get("downstream_fields_missing_everywhere")) >= len(DOWNSTREAM_FIELDS) // 2:
        flags.append("missing_downstream_trajectory_fields")
    if safe_int(values.get("downstream_fields_partial_missing")) > 0:
        flags.append("partial_downstream_trajectory_coverage")
    return flags


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
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def top_loss_rows(rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]], use: str, top_n: int) -> list[dict[str, Any]]:
    raw_by_key = {row_key(row): row for row in raw_rows}
    selected = [
        row for row in rows if row.get("recommended_training_use") == use
    ]
    selected.sort(key=lambda row: safe_float(row.get("realized_loss")), reverse=True)
    output = []
    for row in selected[: max(0, top_n)]:
        raw = raw_by_key.get(row_key(row), {})
        payload = {
            **row,
            "hero_board": raw.get("hero_board"),
            "opponent_board": raw.get("opponent_board"),
            "dead_cards": raw.get("dead_cards"),
            "cards_to_place": raw.get("cards_to_place"),
            "baseline_action": raw.get("baseline_action"),
            "candidate_action": raw.get("candidate_action"),
        }
        output.append(payload)
    return output


def high_local_ev_controls(rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    raw_by_key = {row_key(row): row for row in raw_rows}
    selected = [row for row in rows if row.get("recommended_training_use") == "whole_game_non_loss_control"]
    selected.sort(key=lambda row: safe_float(row.get("local_replay_delta")), reverse=True)
    output = []
    for row in selected[: max(0, top_n)]:
        raw = raw_by_key.get(row_key(row), {})
        output.append(
            {
                **row,
                "hero_board": raw.get("hero_board"),
                "opponent_board": raw.get("opponent_board"),
                "dead_cards": raw.get("dead_cards"),
                "cards_to_place": raw.get("cards_to_place"),
                "baseline_action": raw.get("baseline_action"),
                "candidate_action": raw.get("candidate_action"),
            }
        )
    return output


def write_summary(
    path: Path,
    metrics: list[dict[str, Any]],
    flags: list[str],
    coverage: list[dict[str, Any]],
    coverage_by_group: list[dict[str, Any]],
    trajectory_components: list[dict[str, Any]],
    trajectory_auc: list[dict[str, Any]],
) -> None:
    values = {str(row["metric"]): row["value"] for row in metrics}
    missing_all = [row["field"] for row in coverage if safe_int(row.get("present_rows")) == 0]
    present_count = len(DOWNSTREAM_FIELDS) - len(missing_all)
    mostly_present = present_count >= max(1, len(DOWNSTREAM_FIELDS) // 2)
    lines = [
        "# HU T2 Stage8c Risk Target Gap Analysis",
        "",
        "This diagnostic compares local T2 replay labels with whole-game realized risk labels.",
        "It is not a production, P2, T1, or 50k-teacher approval artifact.",
        f"Primary metric source is `{PRIMARY_METRIC_SOURCE}`. `confirm_delta_mean` and `confirm_delta_z_mean` are `{CONFIRM_DELTA_METRIC_ROLE}` only.",
        "",
        "## Key Counts",
        "",
        f"- rows: `{values.get('deduped_rows', 0)}`",
        f"- whole-game risk-only losses: `{values.get('whole_game_risk_only', 0)}`",
        f"- whole-game non-loss controls: `{values.get('whole_game_non_loss_control', 0)}`",
        f"- local EV hard negatives: `{values.get('local_ev_hard_negative', 0)}`",
        f"- risk-only rows with local positive replay: `{values.get('risk_only_local_positive', 0)}` "
        f"({float(values.get('risk_only_local_positive_share', 0.0)):.3f})",
        f"- risk-only rows with local negative replay: `{values.get('risk_only_local_negative', 0)}`",
        f"- non-loss controls with local negative replay: `{values.get('non_loss_controls_local_negative', 0)}`",
        "",
        "## Diagnostic Flags",
        "",
    ]
    if flags:
        lines.extend(f"- `{flag}`" for flag in flags)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Downstream Field Coverage",
            "",
            f"- fields missing in every row: `{len(missing_all)} / {len(DOWNSTREAM_FIELDS)}`",
            f"- rows with all downstream fields: `{values.get('downstream_trajectory_complete_rows', 0)} / {values.get('deduped_rows', 0)}` "
            f"({float(values.get('downstream_trajectory_complete_share', 0.0)):.3f})",
            f"- fields partially missing: `{values.get('downstream_fields_partial_missing', 0)} / {len(DOWNSTREAM_FIELDS)}`",
        ]
    )
    if missing_all:
        lines.append(f"- missing everywhere: `{', '.join(missing_all)}`")
    else:
        lines.append("- no tracked downstream fields are missing everywhere")
    source_complete = [
        row
        for row in coverage_by_group
        if row.get("group_field") == "source_family" and row.get("field") == "all_downstream_fields"
    ]
    if source_complete:
        lines.extend(
            [
                "",
                "Coverage by source family:",
                "",
                "| source family | rows | complete rows | complete rate |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in sorted(source_complete, key=lambda item: str(item.get("group_value", ""))):
            lines.append(
                "| {source} | {rows} | {present} | {rate:.3f} |".format(
                    source=row.get("group_value", ""),
                    rows=safe_int(row.get("rows")),
                    present=safe_int(row.get("present_rows")),
                    rate=safe_float(row.get("present_rate")),
                )
            )
    top_auc = trajectory_auc[:8]
    top_components = sorted(
        trajectory_components,
        key=lambda row: abs(safe_float(row.get("risk_minus_control_mean"))),
        reverse=True,
    )[:8]
    if mostly_present:
        trajectory_note = (
            "- Most tracked downstream trajectory fields are present in at least part of the dataset. "
            "They help explain whole-game risk labels, but they are not runtime-available inputs for a deployable T2 gate."
        )
    else:
        trajectory_note = (
            "- Many downstream trajectory fields are still missing, so the current risk head is asked to infer "
            "future foul/FL/royalty/scoop outcomes from a single T2 state/action snapshot."
        )
    recommended_additions = []
    if "t3_decision_summary" in missing_all:
        recommended_additions.append("- T3 continuation decision summary")
    if "downstream_override_fired" in missing_all:
        recommended_additions.append("- downstream override fired flags")
    if "paired_future_delta_summary" in missing_all:
        recommended_additions.append("- per-future paired delta summaries for fired decisions")
    if not mostly_present:
        recommended_additions.extend(
            [
                "- post-T2 candidate and baseline boards",
                "- final hero/opponent boards",
                "- hero/opponent foul flags",
                "- hero/opponent FL entry/stay flags",
                "- royalty, line, and scoop deltas",
            ]
        )
    if not recommended_additions:
        recommended_additions.append(
            "- no additional tracked schema fields are missing; generate fresh risk rows with this instrumentation"
        )
    lines.extend(
        [
            "",
            "## Strongest Trajectory Signals",
            "",
            "| field | best AUC | direction | risk mean | control mean |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for row in top_auc:
        lines.append(
            "| {field} | {auc:.4f} | {direction} | {risk:.4f} | {control:.4f} |".format(
                field=row.get("field", ""),
                auc=safe_float(row.get("best_auc")),
                direction=row.get("best_direction", ""),
                risk=safe_float(row.get("risk_mean")),
                control=safe_float(row.get("control_mean")),
            )
        )
    lines.extend(
        [
            "",
            "Largest risk/control mean gaps:",
            "",
            "| field | risk mean | control mean | risk-control |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in top_components:
        lines.append(
            "| {field} | {risk:.4f} | {control:.4f} | {diff:.4f} |".format(
                field=row.get("field", ""),
                risk=safe_float(row.get("risk_mean")),
                control=safe_float(row.get("control_mean")),
                diff=safe_float(row.get("risk_minus_control_mean")),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If most risk-only losses are locally positive, the current local HU feature row is not explaining the loss by local T2 action quality.",
            "- If local-negative rows can also be whole-game non-loss controls, local EV hard-negative labels and whole-game risk labels are not interchangeable.",
            trajectory_note,
            "- Confirm-delta aggregates are runtime feature diagnostics, not performance evidence.",
            "- Trajectory component rows are realized-after-the-fact diagnostics. They can explain which downstream path failed, but they are not directly deployable runtime gate inputs.",
            "- Similar local replay deltas for risk-only losses and non-loss controls indicate that this is a downstream trajectory/risk-separation problem, not just a local EV-label problem.",
            "",
            "Recommended next data additions before another serious risk-head attempt:",
            "",
            *recommended_additions,
            "",
            "Decision: runtime risk integration remains `No-Go`; this artifact is diagnostic only.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = target_paths(args.collection_dir, args.input_jsonl)
    raw_rows, source_rows = load_rows(paths)
    rows = [normalized_row(row) for row in raw_rows]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = metric_rows(rows, raw_rows, source_rows)
    metric_values = {row["metric"]: row["value"] for row in metrics}
    coverage = field_coverage_rows(raw_rows)
    coverage_by_group = field_coverage_by_group_rows(raw_rows)
    flags = diagnostic_flags(metrics)

    write_csv(args.output_dir / "risk_target_gap_metrics.csv", metrics)
    write_csv(args.output_dir / "risk_target_gap_breakdown.csv", breakdown_rows(rows))
    write_csv(args.output_dir / "local_vs_whole_game_matrix.csv", matrix_rows(rows))
    write_csv(args.output_dir / "downstream_field_coverage.csv", coverage)
    write_csv(args.output_dir / "downstream_field_coverage_by_group.csv", coverage_by_group)
    trajectory_components = trajectory_component_rows(rows)
    trajectory_auc = trajectory_feature_auc_rows(rows)
    write_csv(args.output_dir / "trajectory_component_breakdown.csv", trajectory_components)
    write_csv(args.output_dir / "trajectory_feature_auc.csv", trajectory_auc)
    write_csv(args.output_dir / "risk_target_gap_rows.csv", rows)
    write_csv(args.output_dir / "risk_target_gap_sources.csv", source_rows)
    write_jsonl(
        args.output_dir / "risk_only_top_losses.jsonl",
        top_loss_rows(rows, raw_rows, "whole_game_risk_only", args.top_n),
    )
    write_jsonl(
        args.output_dir / "non_loss_high_local_ev_controls.jsonl",
        high_local_ev_controls(rows, raw_rows, args.top_n),
    )
    write_summary(
        args.output_dir / "risk_target_gap_summary.md",
        metrics,
        flags,
        coverage,
        coverage_by_group,
        trajectory_components,
        trajectory_auc,
    )
    manifest = {
        "schema": "hu_turn2_stage8c_risk_target_gap_v1",
        "input_paths": [str(path) for path in paths],
        "output_dir": str(args.output_dir),
        "rows": len(rows),
        "primary_metric_source": PRIMARY_METRIC_SOURCE,
        "confirm_delta_metric_role": CONFIRM_DELTA_METRIC_ROLE,
        "confirm_delta_performance_claim_allowed": CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED,
        "diagnostic_flags": flags,
        "downstream_trajectory_complete_rows": metric_values.get("downstream_trajectory_complete_rows", 0),
        "downstream_trajectory_complete_share": metric_values.get("downstream_trajectory_complete_share", 0.0),
        "downstream_fields_partial_missing": metric_values.get("downstream_fields_partial_missing", 0),
        "partial_downstream_trajectory_coverage": "partial_downstream_trajectory_coverage" in flags,
        "runtime_risk_integration": False,
        "production_p2_fixed": "No-Go",
        "teacher_50k": "No-Go",
        "t1_training": "No-Go",
    }
    (args.output_dir / "risk_target_gap_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
