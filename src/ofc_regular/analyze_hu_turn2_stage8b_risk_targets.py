"""Audit Stage8b TopK loss targets before risk-head training.

This is a readiness/reporting step only. Whole-game realized losses are kept
separate from local T2 EV hard negatives because a whole-hand loss can still be
locally positive at T2.
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
REALIZED_DELTA_METRIC_SOURCE = "realized_whole_game_delta"
CONFIRM_DELTA_METRIC_ROLE = "gate_diagnostic_only"
CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED = False
REPLAY_FIELDS = (
    "hero_board",
    "opponent_board",
    "dead_cards",
    "cards_to_place",
    "baseline_action",
    "candidate_action",
    "candidate_index",
    "baseline_index",
    "local_replay_status",
    "local_replay_action_mapping_status",
    "local_replay_delta",
    "local_replay_delta_se",
)
DOWNSTREAM_TRAJECTORY_FIELDS = (
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


def downstream_complete(row: dict[str, Any]) -> int:
    value = row.get("downstream_trajectory_complete")
    if value not in (None, ""):
        return 1 if safe_int(value) else 0
    present_fields = safe_int(row.get("downstream_trajectory_present_fields"), -1)
    total_fields = safe_int(row.get("downstream_trajectory_total_fields"), -1)
    if present_fields >= 0 and total_fields > 0:
        return int(present_fields == total_fields)
    return int(all(present(row.get(field)) for field in DOWNSTREAM_TRAJECTORY_FIELDS))


def present(value: Any) -> bool:
    return value not in (None, "", [], {})


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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-risk-head-rows", type=int, default=200)
    parser.add_argument("--min-local-ev-hard-negatives", type=int, default=20)
    parser.add_argument("--top-n-losses", type=int, default=50)
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
    paths = [directory / DEFAULT_TARGET_NAME for directory in collection_dirs]
    paths.extend(input_jsonls)
    if not paths:
        raise SystemExit("at least one --collection-dir or --input-jsonl is required")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    return paths


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                row["_audit_source_path"] = str(path)
                rows.append(row)
    return rows


def load_rows(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    source_rows: list[dict[str, Any]] = []
    for path in paths:
        rows = read_jsonl(path)
        source_rows.append({"source_path": str(path), "rows": len(rows)})
        for row in rows:
            current = by_key.get(row_key(row))
            if current is None:
                by_key[row_key(row)] = row
                continue
            if safe_int(row.get("local_replay_future_samples")) > safe_int(current.get("local_replay_future_samples")):
                by_key[row_key(row)] = row
    return sorted(by_key.values(), key=row_key), source_rows


def missing_replay_fields(row: dict[str, Any]) -> list[str]:
    missing = []
    for field in REPLAY_FIELDS:
        value = row.get(field)
        if value in (None, ""):
            missing.append(field)
    return missing


def replay_ready(row: dict[str, Any]) -> bool:
    return (
        not missing_replay_fields(row)
        and str(row.get("local_replay_status", "")).lower() == "ok"
        and str(row.get("local_replay_action_mapping_status", "")).lower() == "ok"
    )


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


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil((q / 100.0) * len(ordered)) - 1))
    return ordered[index]


def normalized_row(row: dict[str, Any]) -> dict[str, Any]:
    missing = missing_replay_fields(row)
    local_delta = safe_float(row.get("local_replay_delta"))
    realized_delta = safe_float(row.get("realized_delta"))
    return {
        "source_log": row.get("source_log", ""),
        "config_id": row.get("config_id", ""),
        "hand_seed": row.get("hand_seed", ""),
        "seat": row.get("seat", ""),
        "seat_swap": row.get("seat_swap", ""),
        "recommended_training_use": recommended_use(row),
        "use_for_local_ev_hard_negative": int(truthy(row.get("use_for_local_ev_hard_negative"))),
        "use_for_whole_game_risk_head": int(truthy(row.get("use_for_whole_game_risk_head"))),
        "requires_local_replay": int(truthy(row.get("requires_local_replay"))),
        "replay_ready": int(replay_ready(row)),
        "downstream_trajectory_complete": downstream_complete(row),
        "downstream_trajectory_present_fields": safe_int(row.get("downstream_trajectory_present_fields")),
        "downstream_trajectory_total_fields": safe_int(row.get("downstream_trajectory_total_fields")),
        "missing_replay_fields": ",".join(missing),
        "realized_delta": realized_delta,
        "realized_loss": max(0.0, -realized_delta),
        "local_replay_bucket": row.get("local_replay_bucket", ""),
        "local_replay_label": row.get("local_replay_label", ""),
        "local_replay_delta": local_delta,
        "local_replay_lcb196": safe_float(row.get("local_replay_lcb196")),
        "local_replay_delta_se": safe_float(row.get("local_replay_delta_se")),
        "confirm_delta": safe_float(row.get("confirm_delta")),
        "confirm_delta_se": safe_float(row.get("confirm_delta_se")),
        "predicted_delta": safe_float(row.get("predicted_delta")),
        "gate_probability": safe_float(row.get("gate_probability")),
        "candidate_ev_rank": safe_int(row.get("candidate_ev_rank"), 9999),
        "candidate_index": safe_int(row.get("candidate_index"), -1),
        "baseline_index": safe_int(row.get("baseline_index"), -1),
        "collection_source_path": row.get("collection_source_path", ""),
        "_audit_source_path": row.get("_audit_source_path", ""),
    }


def group_breakdown(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = [normalized_row(row) for row in rows]
    groups: list[tuple[str, str, list[dict[str, Any]]]] = [("overall", "all", normalized)]
    for field in ("recommended_training_use", "seat", "config_id", "local_replay_bucket", "local_replay_label"):
        for value in sorted({str(row.get(field, "")) for row in normalized}):
            groups.append((field, value, [row for row in normalized if str(row.get(field, "")) == value]))
    output: list[dict[str, Any]] = []
    for field, value, subset in groups:
        realized = [safe_float(row.get("realized_delta")) for row in subset]
        losses = [safe_float(row.get("realized_loss")) for row in subset]
        local = [safe_float(row.get("local_replay_delta")) for row in subset]
        output.append(
            {
                "group_field": field,
                "group_value": value,
                "rows": len(subset),
                "replay_ready_rows": sum(safe_int(row.get("replay_ready")) for row in subset),
                "downstream_trajectory_complete_rows": sum(
                    safe_int(row.get("downstream_trajectory_complete")) for row in subset
                ),
                "local_ev_hard_negative": sum(
                    1 for row in subset if row.get("recommended_training_use") == "local_ev_hard_negative"
                ),
                "whole_game_risk_only": sum(
                    1 for row in subset if row.get("recommended_training_use") == "whole_game_risk_only"
                ),
                "whole_game_non_loss_control": sum(
                    1 for row in subset if row.get("recommended_training_use") == "whole_game_non_loss_control"
                ),
                "requires_local_replay": sum(
                    1 for row in subset if row.get("recommended_training_use") == "requires_local_replay"
                ),
                "realized_delta_mean": mean(realized),
                "realized_loss_mean": mean(losses),
                "realized_loss_p95": percentile(losses, 95),
                "realized_loss_max": max(losses, default=0.0),
                "local_replay_delta_mean": mean(local),
                "predicted_delta_mean": mean([safe_float(row.get("predicted_delta")) for row in subset]),
                "confirm_delta_mean": mean([safe_float(row.get("confirm_delta")) for row in subset]),
                "confirm_delta_metric_role": CONFIRM_DELTA_METRIC_ROLE,
                "confirm_delta_performance_claim_allowed": CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED,
                "realized_delta_metric_source": REALIZED_DELTA_METRIC_SOURCE,
            }
        )
    return output


def summary_metrics(rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = [normalized_row(row) for row in rows]
    use_counts = Counter(str(row.get("recommended_training_use")) for row in normalized)
    seat_counts = Counter(str(row.get("seat")) for row in normalized)
    missing_counter: Counter[str] = Counter()
    for row in rows:
        missing_counter.update(missing_replay_fields(row))
    metrics: list[dict[str, Any]] = [
        {"metric": "input_paths", "value": len(source_rows)},
        {"metric": "input_rows", "value": sum(safe_int(row.get("rows")) for row in source_rows)},
        {"metric": "deduped_rows", "value": len(rows)},
        {"metric": "realized_delta_metric_source", "value": REALIZED_DELTA_METRIC_SOURCE},
        {"metric": "confirm_delta_metric_role", "value": CONFIRM_DELTA_METRIC_ROLE},
        {
            "metric": "confirm_delta_performance_claim_allowed",
            "value": CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED,
        },
        {"metric": "replay_ready_rows", "value": sum(replay_ready(row) for row in rows)},
        {"metric": "missing_replay_field_rows", "value": sum(1 for row in rows if missing_replay_fields(row))},
        {"metric": "downstream_trajectory_complete_rows", "value": sum(downstream_complete(row) for row in rows)},
        {
            "metric": "downstream_trajectory_incomplete_rows",
            "value": sum(1 for row in rows if not downstream_complete(row)),
        },
        {"metric": "realized_delta_mean", "value": mean([safe_float(row.get("realized_delta")) for row in rows])},
        {"metric": "local_replay_delta_mean", "value": mean([safe_float(row.get("local_replay_delta")) for row in rows])},
    ]
    for key, count in sorted(use_counts.items()):
        metrics.append({"metric": f"recommended_use.{key}", "value": count})
    for key, count in sorted(seat_counts.items()):
        metrics.append({"metric": f"seat.{key}", "value": count})
    for key, count in sorted(missing_counter.items()):
        metrics.append({"metric": f"missing_field.{key}", "value": count})
    return metrics


def readiness(
    rows: list[dict[str, Any]],
    *,
    min_risk_head_rows: int,
    min_local_ev_hard_negatives: int,
) -> dict[str, Any]:
    normalized = [normalized_row(row) for row in rows]
    risk_only = [row for row in normalized if row["recommended_training_use"] == "whole_game_risk_only"]
    local_hard = [row for row in normalized if row["recommended_training_use"] == "local_ev_hard_negative"]
    non_loss_controls = [
        row for row in normalized if row["recommended_training_use"] == "whole_game_non_loss_control"
    ]
    local_positive_controls = [
        row
        for row in normalized
        if row["local_replay_label"] == "positive" and row["local_replay_bucket"] == "local_positive_lcb"
    ]
    seats = {str(row.get("seat")) for row in normalized if str(row.get("seat"))}
    replay_ready_rows = sum(safe_int(row.get("replay_ready")) for row in normalized)
    missing_replay_rows = len(normalized) - replay_ready_rows
    risk_trainable = risk_only + non_loss_controls
    local_trainable = local_hard + local_positive_controls
    trajectory_trainable = risk_only + non_loss_controls + local_hard
    risk_missing_replay_rows = sum(1 for row in risk_trainable if not safe_int(row.get("replay_ready")))
    local_missing_replay_rows = sum(1 for row in local_trainable if not safe_int(row.get("replay_ready")))
    trajectory_complete_rows = sum(safe_int(row.get("downstream_trajectory_complete")) for row in trajectory_trainable)
    trajectory_incomplete_rows = len(trajectory_trainable) - trajectory_complete_rows
    blockers: list[str] = []
    if risk_missing_replay_rows:
        blockers.append("missing_replay_fields")
    if len(risk_only) < min_risk_head_rows:
        blockers.append("risk_only_rows_lt_min")
    if not non_loss_controls:
        blockers.append("missing_non_loss_control_rows")
    if len(seats) < 2:
        blockers.append("single_seat_only")
    local_blockers: list[str] = []
    if local_missing_replay_rows:
        local_blockers.append("missing_replay_fields")
    if len(local_hard) < min_local_ev_hard_negatives:
        local_blockers.append("local_ev_hard_negatives_lt_min")
    if not local_positive_controls:
        local_blockers.append("missing_local_ev_positive_controls")
    local_ready = not local_blockers
    return {
        "schema": "hu_turn2_stage8b_risk_target_readiness_v1",
        "rows": len(normalized),
        "risk_only_rows": len(risk_only),
        "local_ev_hard_negatives": len(local_hard),
        "local_ev_positive_lcb_controls": len(local_positive_controls),
        "non_loss_control_rows": len(non_loss_controls),
        "seats": sorted(seats),
        "replay_ready_rows": replay_ready_rows,
        "missing_replay_rows": missing_replay_rows,
        "risk_trainable_rows": len(risk_trainable),
        "risk_trainable_missing_replay_rows": risk_missing_replay_rows,
        "local_ev_trainable_rows": len(local_trainable),
        "local_ev_trainable_missing_replay_rows": local_missing_replay_rows,
        "trajectory_component_analysis_rows": len(trajectory_trainable),
        "trajectory_component_complete_rows": trajectory_complete_rows,
        "trajectory_component_incomplete_rows": trajectory_incomplete_rows,
        "trajectory_component_analysis_ready": trajectory_incomplete_rows == 0 and bool(trajectory_trainable),
        "min_risk_head_rows": min_risk_head_rows,
        "min_local_ev_hard_negatives": min_local_ev_hard_negatives,
        "risk_head_training_ready": len(blockers) == 0,
        "local_ev_hard_negative_training_ready": local_ready,
        "blockers": blockers,
        "risk_head_blockers": blockers,
        "local_ev_hard_negative_blockers": local_blockers,
        "production_p2_fixed": "No-Go",
        "t1_training": "No-Go",
        "teacher_50k": "No-Go",
    }


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
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_summary(path: Path, metrics: list[dict[str, Any]], ready: dict[str, Any]) -> None:
    lines = [
        "# HU T2 Stage8b Risk Target Audit",
        "",
        "This audit checks whether TopK whole-game loss targets are ready for risk-head training.",
        "It does not train a model and it does not approve production, P2 fixed status, T1, or 50k teacher generation.",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for row in metrics:
        lines.append(f"| {row['metric']} | {row['value']} |")
    lines.extend(
        [
            "",
            "## Readiness",
            "",
            f"- risk_head_training_ready: `{ready['risk_head_training_ready']}`",
            f"- local_ev_hard_negative_training_ready: `{ready['local_ev_hard_negative_training_ready']}`",
            f"- trajectory_component_analysis_ready: `{ready['trajectory_component_analysis_ready']}`",
            f"- risk_head_blockers: `{', '.join(ready['risk_head_blockers']) if ready['risk_head_blockers'] else 'none'}`",
            f"- local_ev_hard_negative_blockers: `{', '.join(ready['local_ev_hard_negative_blockers']) if ready['local_ev_hard_negative_blockers'] else 'none'}`",
            f"- trajectory_component_complete_rows: `{ready['trajectory_component_complete_rows']} / {ready['trajectory_component_analysis_rows']}`",
            "",
            "Interpretation:",
            "",
            "- `local_ev_hard_negative` rows may be used only for local EV/gate hard-negative training.",
            "- `whole_game_risk_only` rows require a separate downstream whole-game risk/counterfactual head.",
            "- Realized whole-game losses must not be mixed into safe-LCB/local EV labels unless local replay says they are local EV negatives.",
            "- `confirm_delta_mean` is a gate diagnostic only; do not cite it as realized whole-game performance.",
            "- `trajectory_component_analysis_ready` is separate from training readiness; incomplete rows can still train risk/local labels but cannot explain foul/FL/royalty/scoop loss causes.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = target_paths(args.collection_dir, args.input_jsonl)
    rows, source_rows = load_rows(paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    normalized = [normalized_row(row) for row in rows]
    metrics = summary_metrics(rows, source_rows)
    ready = readiness(
        rows,
        min_risk_head_rows=args.min_risk_head_rows,
        min_local_ev_hard_negatives=args.min_local_ev_hard_negatives,
    )
    losses = sorted(normalized, key=lambda row: safe_float(row.get("realized_loss")), reverse=True)[
        : max(0, args.top_n_losses)
    ]

    write_csv(args.output_dir / "risk_target_summary.csv", metrics)
    write_csv(args.output_dir / "risk_target_breakdown.csv", group_breakdown(rows))
    write_csv(args.output_dir / "risk_target_rows.csv", normalized)
    write_jsonl(args.output_dir / "risk_target_top_losses.jsonl", losses)
    write_summary(args.output_dir / "risk_target_summary.md", metrics, ready)
    (args.output_dir / "risk_target_readiness.json").write_text(
        json.dumps(ready, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "hu_turn2_stage8b_risk_target_audit_v1",
        "input_paths": [str(path) for path in paths],
        "output_dir": str(args.output_dir),
        "rows": len(rows),
        "realized_delta_metric_source": REALIZED_DELTA_METRIC_SOURCE,
        "confirm_delta_metric_role": CONFIRM_DELTA_METRIC_ROLE,
        "confirm_delta_performance_claim_allowed": CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED,
        "risk_head_training_ready": ready["risk_head_training_ready"],
        "local_ev_hard_negative_training_ready": ready["local_ev_hard_negative_training_ready"],
        "trajectory_component_analysis_ready": ready["trajectory_component_analysis_ready"],
        "trajectory_component_analysis_rows": ready["trajectory_component_analysis_rows"],
        "trajectory_component_complete_rows": ready["trajectory_component_complete_rows"],
        "trajectory_component_incomplete_rows": ready["trajectory_component_incomplete_rows"],
        "blockers": ready["blockers"],
    }
    (args.output_dir / "risk_target_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
