"""Build whole-game counterfactual risk targets from Stage8b TopK logs.

The local T2 replay label and the realized whole-game counterfactual label are
not interchangeable. This tool joins fired runtime decision logs with optional
local T2 replay summaries, then separates:

- local T2 EV hard negatives, which can train the local EV/gate target, from
- whole-game realized losses, which need a separate risk/counterfactual target.

By default it keeps historical loss-only behavior. Use
``--include-non-loss-controls`` after all-fired replay when preparing controls
for a separate whole-game risk/counterfactual head.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT_DIR = Path("outputs/evals/hu_turn2_stage8b_counterfactual_loss_targets")
TOPK_HARD_NEGATIVE_PACK_SCHEMA = "hu_turn2_stage8b_topk_hard_negative_v1"
TRAJECTORY_SUFFIX_FIELDS = (
    "final_board_hero",
    "final_board_opponent",
    "hero_foul",
    "opponent_foul",
    "hero_royalty",
    "opponent_royalty",
    "royalty_delta",
    "hero_fl_entry",
    "hero_fl_card_count",
    "hero_fl_entry_type",
    "hero_fl_value",
    "hero_fl_stay",
    "opponent_fl_entry",
    "opponent_fl_card_count",
    "opponent_fl_entry_type",
    "opponent_fl_value",
    "opponent_fl_stay",
    "fl_delta",
    "line_results",
    "line_score_delta",
    "scoop_delta",
    "foul_delta",
    "terminal_score",
)
TRAJECTORY_DELTA_FIELDS = (
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
)
PAIRED_FUTURE_SUMMARY_FIELDS = (
    "stage_a_paired_delta_summary",
    "confirm_paired_delta_summary",
    "paired_future_delta_summary",
    "paired_future_delta_source",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decision-log",
        type=Path,
        action="append",
        required=True,
        help="Runtime decision JSONL from evaluate_hu_turn2_stage8b_topk_mc_rerank. Can be repeated.",
    )
    parser.add_argument(
        "--local-replay-summary",
        type=Path,
        action="append",
        default=[],
        help="CSV from replay_hu_turn2_stage8b_topk_hard_negatives. Can be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--realized-loss-threshold", type=float, default=0.0)
    parser.add_argument("--local-positive-lcb-threshold", type=float, default=0.0)
    parser.add_argument(
        "--include-non-loss-controls",
        action="store_true",
        help=(
            "Also emit fired rows with realized_delta >= threshold as whole-game "
            "non-loss controls for a separate risk/counterfactual head. Default "
            "keeps the historical loss-only behavior."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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


def present(value: Any) -> bool:
    return value not in (None, "", [], {})


def row_key(
    *,
    source_log: Any,
    config_id: Any,
    hand_seed: Any,
    seat: Any,
    candidate_index: Any,
    baseline_index: Any,
) -> tuple[str, str, str, str, str, str]:
    return (
        normalize_path_key(source_log),
        str(config_id or ""),
        str(hand_seed or ""),
        str(seat or ""),
        str(safe_int(candidate_index, -1)),
        str(safe_int(baseline_index, -1)),
    )


def normalize_path_key(value: Any) -> str:
    return str(value or "").replace("/", "\\").lower()


def runtime_decision_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return row_key(
        source_log=source_log_for_replay_join(row),
        config_id=row.get("config_id", ""),
        hand_seed=row.get("hand_seed", ""),
        seat=row.get("seat", ""),
        candidate_index=candidate_index_for_target(row),
        baseline_index=baseline_index_for_target(row),
    )


def runtime_decision_loose_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return row_key(
        source_log="",
        config_id="",
        hand_seed=row.get("hand_seed", ""),
        seat=row.get("seat", ""),
        candidate_index=candidate_index_for_target(row),
        baseline_index=baseline_index_for_target(row),
    )


def replay_summary_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return row_key(
        source_log=row.get("source_log", ""),
        config_id=row.get("source_config_id", ""),
        hand_seed=row.get("hand_seed", ""),
        seat=row.get("seat", ""),
        candidate_index=row.get("candidate_index", -1),
        baseline_index=row.get("logged_baseline_index", -1),
    )


def replay_summary_loose_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return row_key(
        source_log="",
        config_id="",
        hand_seed=row.get("hand_seed", ""),
        seat=row.get("seat", ""),
        candidate_index=row.get("candidate_index", -1),
        baseline_index=row.get("logged_baseline_index", -1),
    )


def build_replay_index(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str, str, str, str, str], dict[str, Any]]:
    output: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        for key in (replay_summary_key(row), replay_summary_loose_key(row)):
            current = output.get(key)
            if current is None or safe_int(row.get("future_samples")) > safe_int(current.get("future_samples")):
                output[key] = row
    return output


def local_replay_bucket(row: dict[str, Any] | None, *, positive_lcb_threshold: float = 0.0) -> str:
    if row is None:
        return "missing"
    if row.get("status") != "ok":
        return "failed"
    delta = safe_float(row.get("delta_for_label"))
    lcb = safe_float(row.get("replay_delta_lcb196"))
    hard = safe_int(row.get("hard_negative_label"))
    if hard:
        return "local_negative"
    if lcb > positive_lcb_threshold:
        return "local_positive_lcb"
    if delta > 0.0:
        return "local_positive_gray"
    if delta < 0.0:
        return "local_negative_delta"
    return "local_neutral"


def target_use_for(
    realized_delta: float,
    replay_bucket: str,
    *,
    loss_threshold: float = 0.0,
    include_non_loss_controls: bool = False,
) -> str:
    if realized_delta >= loss_threshold:
        if include_non_loss_controls:
            return "whole_game_non_loss_control"
        return "not_realized_loss"
    if replay_bucket in {"local_negative", "local_negative_delta"}:
        return "local_ev_hard_negative"
    if replay_bucket in {"local_positive_lcb", "local_positive_gray", "local_neutral"}:
        return "whole_game_risk_only"
    if replay_bucket == "missing":
        return "requires_local_replay"
    return "inspect_before_training"


def _copy_if_present(output: dict[str, Any], key: str, value: Any) -> None:
    if value not in (None, ""):
        output[key] = value


def is_topk_pack_row(row: dict[str, Any]) -> bool:
    return row.get("schema") == TOPK_HARD_NEGATIVE_PACK_SCHEMA


def source_log_for_replay_join(row: dict[str, Any]) -> Any:
    if is_topk_pack_row(row):
        return row.get("source_log", "")
    return row.get("_source_log", "")


def candidate_index_for_target(row: dict[str, Any]) -> int:
    if is_topk_pack_row(row):
        return safe_int(row.get("candidate_action_index", row.get("rerank_best_index")), -1)
    return safe_int(row.get("final_action_index", row.get("rerank_best_index")), -1)


def baseline_index_for_target(row: dict[str, Any]) -> int:
    return safe_int(row.get("baseline_action_index"), -1)


def realized_delta_for_target(row: dict[str, Any]) -> float:
    if is_topk_pack_row(row):
        return safe_float(row.get("realized_delta"))
    return safe_float(row.get("realized_candidate_seat_delta"))


def is_target_candidate_row(row: dict[str, Any]) -> bool:
    if is_topk_pack_row(row):
        return row.get("realized_delta") not in (None, "")
    return truthy(row.get("override_fired")) and truthy(row.get("realized_delta_valid"))


def confirm_delta_for_target(row: dict[str, Any]) -> float:
    return safe_float(row.get("rerank_delta", row.get("confirm_delta")))


def confirm_delta_se_for_target(row: dict[str, Any]) -> float:
    return safe_float(row.get("rerank_delta_se", row.get("confirm_delta_se")))


def confirm_count_for_target(row: dict[str, Any]) -> int:
    return safe_int(row.get("confirm_delta_count", row.get("confirm_count")), 0)


def candidate_action_for_target(row: dict[str, Any]) -> Any:
    return row.get("final_action", row.get("candidate_action"))


def post_t2_candidate_board_for_target(row: dict[str, Any]) -> Any:
    if row.get("post_t2_candidate_board") not in (None, ""):
        return row.get("post_t2_candidate_board")
    candidate_action = row.get("candidate_action")
    return candidate_action.get("next_board") if isinstance(candidate_action, dict) else None


def post_t2_baseline_board_for_target(row: dict[str, Any]) -> Any:
    if row.get("post_t2_baseline_board") not in (None, ""):
        return row.get("post_t2_baseline_board")
    baseline_action = row.get("baseline_action")
    return baseline_action.get("next_board") if isinstance(baseline_action, dict) else None


def add_trajectory_fields(output: dict[str, Any], row: dict[str, Any]) -> None:
    _copy_if_present(output, "post_t2_candidate_board", post_t2_candidate_board_for_target(row))
    _copy_if_present(output, "post_t2_baseline_board", post_t2_baseline_board_for_target(row))
    _copy_if_present(output, "post_t2_rerank_best_board", row.get("post_t2_rerank_best_board"))
    _copy_if_present(output, "t3_decision_summary", row.get("t3_decision_summary"))
    _copy_if_present(output, "downstream_override_fired", row.get("downstream_override_fired"))
    for suffix in TRAJECTORY_SUFFIX_FIELDS:
        candidate_key = f"candidate_{suffix}"
        baseline_key = f"baseline_{suffix}"
        candidate_value = row.get(candidate_key)
        baseline_value = row.get(baseline_key)
        _copy_if_present(output, suffix, candidate_value)
        _copy_if_present(output, candidate_key, candidate_value)
        _copy_if_present(output, baseline_key, baseline_value)
    for key in TRAJECTORY_DELTA_FIELDS:
        _copy_if_present(output, key, row.get(key))
    for key in PAIRED_FUTURE_SUMMARY_FIELDS:
        _copy_if_present(output, key, row.get(key))


def add_downstream_coverage_fields(output: dict[str, Any]) -> None:
    present_count = sum(1 for field in DOWNSTREAM_TRAJECTORY_FIELDS if present(output.get(field)))
    output["downstream_trajectory_present_fields"] = present_count
    output["downstream_trajectory_total_fields"] = len(DOWNSTREAM_TRAJECTORY_FIELDS)
    output["downstream_trajectory_complete"] = int(present_count == len(DOWNSTREAM_TRAJECTORY_FIELDS))


def target_row(
    row: dict[str, Any],
    *,
    replay_index: dict[tuple[str, str, str, str, str, str], dict[str, Any]],
    loss_threshold: float,
    local_positive_lcb_threshold: float,
    include_non_loss_controls: bool = False,
) -> dict[str, Any] | None:
    if not is_target_candidate_row(row):
        return None
    realized_delta = realized_delta_for_target(row)
    if realized_delta >= loss_threshold and not include_non_loss_controls:
        return None
    replay = replay_index.get(runtime_decision_key(row)) or replay_index.get(runtime_decision_loose_key(row))
    replay_bucket = local_replay_bucket(replay, positive_lcb_threshold=local_positive_lcb_threshold)
    recommended_use = target_use_for(
        realized_delta,
        replay_bucket,
        loss_threshold=loss_threshold,
        include_non_loss_controls=include_non_loss_controls,
    )
    output = {
        "schema": "hu_turn2_stage8b_counterfactual_loss_target_v1",
        "source_log": row.get("source_log", row.get("_source_log", "")),
        "target_source_path": row.get("_source_log", ""),
        "config_id": row.get("config_id", ""),
        "hand_seed": row.get("hand_seed", ""),
        "hand_id": row.get("hand_id", ""),
        "game_id": row.get("game_id", ""),
        "seat": row.get("seat", ""),
        "seat_swap": row.get("seat_swap", ""),
        "realized_delta": realized_delta,
        "realized_loss_label": int(realized_delta < loss_threshold),
        "realized_delta_basis": row.get("realized_delta_basis", "topk_replay_pack" if is_topk_pack_row(row) else ""),
        "candidate_index": candidate_index_for_target(row),
        "baseline_index": baseline_index_for_target(row),
        "predicted_delta": safe_float(row.get("predicted_delta")),
        "gate_probability": safe_float(row.get("gate_probability")),
        "candidate_ev_rank": safe_int(row.get("candidate_ev_rank"), -1),
        "confirm_delta": confirm_delta_for_target(row),
        "confirm_delta_se": confirm_delta_se_for_target(row),
        "confirm_delta_count": confirm_count_for_target(row),
        "local_replay_bucket": replay_bucket,
        "recommended_training_use": recommended_use,
        "use_for_local_ev_hard_negative": int(recommended_use == "local_ev_hard_negative"),
        "use_for_whole_game_risk_head": int(
            recommended_use in {"whole_game_risk_only", "whole_game_non_loss_control"}
        ),
        "whole_game_risk_label": int(realized_delta < loss_threshold)
        if recommended_use in {"whole_game_risk_only", "whole_game_non_loss_control"}
        else "",
        "requires_local_replay": int(recommended_use == "requires_local_replay"),
        "hero_board": row.get("hero_board"),
        "opponent_board": row.get("opponent_board"),
        "dead_cards": row.get("dead_cards"),
        "cards_to_place": row.get("cards_to_place"),
        "baseline_action": row.get("baseline_action"),
        "candidate_action": candidate_action_for_target(row),
    }
    add_trajectory_fields(output, row)
    add_downstream_coverage_fields(output)
    if replay is not None:
        output.update(
            {
                "local_replay_status": replay.get("status", ""),
                "local_replay_future_samples": safe_int(replay.get("future_samples"), 0),
                "local_replay_delta": safe_float(replay.get("delta_for_label")),
                "local_replay_delta_se": safe_float(replay.get("delta_standard_error_for_label")),
                "local_replay_lcb196": safe_float(replay.get("replay_delta_lcb196")),
                "local_replay_label": replay.get("safe_lcb196_label", ""),
                "local_replay_action_mapping_status": replay.get("action_mapping_status", ""),
            }
        )
    return output


def target_rows(
    decision_logs: Iterable[tuple[Path, list[dict[str, Any]]]],
    *,
    replay_rows: Iterable[dict[str, Any]] = (),
    loss_threshold: float = 0.0,
    local_positive_lcb_threshold: float = 0.0,
    include_non_loss_controls: bool = False,
) -> list[dict[str, Any]]:
    replay_index = build_replay_index(replay_rows)
    output: list[dict[str, Any]] = []
    for path, rows in decision_logs:
        for row in rows:
            row_with_source = dict(row)
            row_with_source["_source_log"] = str(path)
            target = target_row(
                row_with_source,
                replay_index=replay_index,
                loss_threshold=loss_threshold,
                local_positive_lcb_threshold=local_positive_lcb_threshold,
                include_non_loss_controls=include_non_loss_controls,
            )
            if target is not None:
                output.append(target)
    return output


def summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(str(row.get("recommended_training_use", "")) for row in rows)
    bucket_counts = Counter(str(row.get("local_replay_bucket", "")) for row in rows)
    output = [
        {"metric": "target_rows", "value": len(rows)},
        {"metric": "realized_delta_sum", "value": sum(safe_float(row.get("realized_delta")) for row in rows)},
        {
            "metric": "realized_delta_mean",
            "value": sum(safe_float(row.get("realized_delta")) for row in rows) / len(rows) if rows else 0.0,
        },
        {
            "metric": "downstream_trajectory_complete_rows",
            "value": sum(safe_int(row.get("downstream_trajectory_complete")) for row in rows),
        },
        {
            "metric": "downstream_trajectory_incomplete_rows",
            "value": sum(1 for row in rows if not safe_int(row.get("downstream_trajectory_complete"))),
        },
    ]
    for key, value in sorted(counts.items()):
        output.append({"metric": f"recommended_use.{key}", "value": value})
    for key, value in sorted(bucket_counts.items()):
        output.append({"metric": f"local_replay_bucket.{key}", "value": value})
    return output


def downstream_coverage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups: list[tuple[str, str, list[dict[str, Any]]]] = [("overall", "all", rows)]
    for field in ("realized_delta_basis", "recommended_training_use"):
        for value in sorted({str(row.get(field, "")) for row in rows}):
            groups.append((field, value, [row for row in rows if str(row.get(field, "")) == value]))
    for group_field, group_value, subset in groups:
        total = len(subset)
        complete_count = sum(safe_int(row.get("downstream_trajectory_complete")) for row in subset)
        output.append(
            {
                "group_field": group_field,
                "group_value": group_value,
                "field": "all_downstream_trajectory_fields",
                "rows": total,
                "present_rows": complete_count,
                "missing_rows": total - complete_count,
                "present_rate": complete_count / total if total else 0.0,
            }
        )
        for field in DOWNSTREAM_TRAJECTORY_FIELDS:
            present_count = sum(1 for row in subset if present(row.get(field)))
            output.append(
                {
                    "group_field": group_field,
                    "group_value": group_value,
                    "field": field,
                    "rows": total,
                    "present_rows": present_count,
                    "missing_rows": total - present_count,
                    "present_rate": present_count / total if total else 0.0,
                }
            )
    return output


def write_summary_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    summary = summary_rows(rows)
    values = {str(row["metric"]): row["value"] for row in summary}
    lines = [
        "# HU T2 Stage8b Counterfactual Risk Targets",
        "",
        "These rows are fired TopK decisions with realized whole-game labels. Realized losses are not automatically local T2 EV hard negatives, and non-loss controls are only for a separate risk/counterfactual head.",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for row in summary:
        lines.append(f"| {row['metric']} | {row['value']} |")
    complete = safe_int(values.get("downstream_trajectory_complete_rows"))
    total = len(rows)
    lines.extend(
        [
            "",
            "## Downstream Trajectory Coverage",
            "",
            f"- complete rows: `{complete} / {total}`",
            "- Incomplete rows are valid for local replay labels, but not for foul/FL/royalty/scoop trajectory-cause analysis.",
            "",
            "## Use Rules",
            "",
            "- `local_ev_hard_negative`: can be used as local EV/gate hard negative after feature generation.",
            "- `whole_game_risk_only`: local replay is non-negative; use only for a separate whole-game risk/counterfactual head or audit.",
            "- `whole_game_non_loss_control`: non-loss fired row for a separate whole-game risk/counterfactual head; not a local EV label.",
            "- `requires_local_replay`: do not train from it until local replay has been generated under the current objective.",
            "- This artifact does not approve T2 production, P2 fixed status, T1, or 50k teacher generation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    decision_logs = [(path, read_jsonl(path)) for path in args.decision_log]
    replay_rows: list[dict[str, Any]] = []
    for path in args.local_replay_summary:
        replay_rows.extend(read_csv(path))
    rows = target_rows(
        decision_logs,
        replay_rows=replay_rows,
        loss_threshold=args.realized_loss_threshold,
        local_positive_lcb_threshold=args.local_positive_lcb_threshold,
        include_non_loss_controls=args.include_non_loss_controls,
    )
    mismatch_rows = [
        row
        for row in rows
        if row.get("recommended_training_use") in {"whole_game_risk_only", "requires_local_replay", "inspect_before_training"}
    ]
    write_jsonl(args.output_dir / "topk_counterfactual_loss_targets.jsonl", rows)
    write_csv(args.output_dir / "topk_counterfactual_loss_targets.csv", rows)
    write_csv(args.output_dir / "topk_local_vs_realized_mismatch.csv", mismatch_rows)
    write_csv(args.output_dir / "topk_counterfactual_loss_summary.csv", summary_rows(rows))
    write_csv(args.output_dir / "topk_counterfactual_downstream_coverage.csv", downstream_coverage_rows(rows))
    write_summary_markdown(args.output_dir / "topk_counterfactual_loss_summary.md", rows)
    downstream_complete_rows = sum(safe_int(row.get("downstream_trajectory_complete")) for row in rows)
    downstream_incomplete_rows = len(rows) - downstream_complete_rows
    manifest = {
        "decision_logs": [str(path) for path, _ in decision_logs],
        "local_replay_summaries": [str(path) for path in args.local_replay_summary],
        "rows": len(rows),
        "mismatch_rows": len(mismatch_rows),
        "downstream_trajectory_complete_rows": downstream_complete_rows,
        "downstream_trajectory_incomplete_rows": downstream_incomplete_rows,
        "downstream_trajectory_complete_rate": downstream_complete_rows / len(rows) if rows else 0.0,
        "include_non_loss_controls": bool(args.include_non_loss_controls),
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "topk_counterfactual_loss_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
