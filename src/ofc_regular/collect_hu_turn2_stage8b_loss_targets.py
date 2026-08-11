"""Collect Stage8b TopK counterfactual loss targets across runs.

This collector intentionally keeps local T2 EV hard negatives separate from
whole-game realized-risk rows. A realized whole-game loss is not automatically
a local EV/gate negative; local replay evidence decides that split.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT_DIR = Path("outputs/evals/hu_turn2_stage8b_loss_target_collection")
DEFAULT_TARGET_NAME = "topk_counterfactual_loss_targets.jsonl"
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
        "--input-dir",
        type=Path,
        action="append",
        default=[],
        help=f"Directory containing {DEFAULT_TARGET_NAME}. Can be repeated.",
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        action="append",
        default=[],
        help="Explicit counterfactual loss target JSONL. Can be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def present(value: Any) -> bool:
    return value not in (None, "", [], {})


def downstream_complete(row: dict[str, Any]) -> int:
    value = row.get("downstream_trajectory_complete")
    if value not in (None, ""):
        return 1 if safe_int(value) else 0
    present_fields = safe_int(row.get("downstream_trajectory_present_fields"), -1)
    total_fields = safe_int(row.get("downstream_trajectory_total_fields"), -1)
    if present_fields >= 0 and total_fields > 0:
        return int(present_fields == total_fields)
    return int(all(present(row.get(field)) for field in DOWNSTREAM_TRAJECTORY_FIELDS))


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


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


def target_paths(input_dirs: Iterable[Path], input_jsonls: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    for directory in input_dirs:
        if not directory.exists():
            raise FileNotFoundError(directory)
        paths.append(directory / DEFAULT_TARGET_NAME)
    paths.extend(input_jsonls)
    return paths


def prefer_row(current: dict[str, Any] | None, candidate: dict[str, Any]) -> dict[str, Any]:
    if current is None:
        return candidate
    current_samples = safe_int(current.get("local_replay_future_samples"))
    candidate_samples = safe_int(candidate.get("local_replay_future_samples"))
    if candidate_samples != current_samples:
        return candidate if candidate_samples > current_samples else current
    current_use = str(current.get("recommended_training_use", ""))
    candidate_use = str(candidate.get("recommended_training_use", ""))
    if current_use != candidate_use and candidate_use == "local_ev_hard_negative":
        return candidate
    return current


def collect_rows(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    source_rows: list[dict[str, Any]] = []
    for path in paths:
        rows = read_jsonl(path)
        source_rows.append({"source_path": str(path), "rows": len(rows)})
        for row in rows:
            payload = dict(row)
            payload["collection_source_path"] = str(path)
            key = row_key(payload)
            by_key[key] = prefer_row(by_key.get(key), payload)
    return sorted(by_key.values(), key=row_key), source_rows


def split_rows(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output = {
        "local_ev_hard_negative": [],
        "whole_game_risk_only": [],
        "whole_game_non_loss_control": [],
        "requires_local_replay": [],
        "inspect_before_training": [],
        "other": [],
    }
    for row in rows:
        use = str(row.get("recommended_training_use", ""))
        output.get(use, output["other"]).append(row)
    return output


def summary_rows(rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    use_counts = Counter(str(row.get("recommended_training_use", "")) for row in rows)
    bucket_counts = Counter(str(row.get("local_replay_bucket", "")) for row in rows)
    complete_rows = sum(downstream_complete(row) for row in rows)
    output: list[dict[str, Any]] = [
        {"metric": "input_paths", "value": len(source_rows)},
        {"metric": "input_rows", "value": sum(safe_int(row.get("rows")) for row in source_rows)},
        {"metric": "deduped_rows", "value": len(rows)},
        {"metric": "realized_delta_sum", "value": sum(safe_float(row.get("realized_delta")) for row in rows)},
        {
            "metric": "realized_delta_mean",
            "value": sum(safe_float(row.get("realized_delta")) for row in rows) / len(rows) if rows else 0.0,
        },
        {"metric": "downstream_trajectory_complete_rows", "value": complete_rows},
        {"metric": "downstream_trajectory_incomplete_rows", "value": len(rows) - complete_rows},
    ]
    for key, count in sorted(use_counts.items()):
        output.append({"metric": f"recommended_use.{key}", "value": count})
    for key, count in sorted(bucket_counts.items()):
        output.append({"metric": f"local_replay_bucket.{key}", "value": count})
    return output


def downstream_coverage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups: list[tuple[str, str, list[dict[str, Any]]]] = [("overall", "all", rows)]
    for field in ("recommended_training_use", "collection_source_path"):
        for value in sorted({str(row.get(field, "")) for row in rows}):
            groups.append((field, value, [row for row in rows if str(row.get(field, "")) == value]))
    for group_field, group_value, subset in groups:
        complete_rows = sum(downstream_complete(row) for row in subset)
        output.append(
            {
                "group_field": group_field,
                "group_value": group_value,
                "rows": len(subset),
                "downstream_trajectory_complete_rows": complete_rows,
                "downstream_trajectory_incomplete_rows": len(subset) - complete_rows,
                "downstream_trajectory_complete_rate": complete_rows / len(subset) if subset else 0.0,
            }
        )
    return output


def write_summary_markdown(path: Path, rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    values = {str(row["metric"]): row["value"] for row in summary}
    lines = [
        "# HU T2 Stage8b Loss Target Collection",
        "",
        "This artifact merges replay-ready TopK counterfactual loss targets across runs.",
        "",
        "Use rules:",
        "",
        "- `topk_local_ev_hard_negatives.*` may feed local EV/gate hard-negative feature generation.",
        "- `topk_whole_game_risk_only.*` must not be used as local EV/gate negatives.",
        "- `topk_whole_game_non_loss_controls.*` are controls for a separate whole-game risk/counterfactual head.",
        "- `requires_local_replay` rows need current-objective local replay before training.",
        "- This artifact does not approve T2 production, P2 fixed status, T1, or 50k teacher generation.",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for row in summary:
        lines.append(f"| {row['metric']} | {row['value']} |")
    if rows:
        local_count = sum(str(row.get("recommended_training_use")) == "local_ev_hard_negative" for row in rows)
        risk_count = sum(str(row.get("recommended_training_use")) == "whole_game_risk_only" for row in rows)
        control_count = sum(
            str(row.get("recommended_training_use")) == "whole_game_non_loss_control" for row in rows
        )
        lines.extend(
            [
                "",
                "Downstream trajectory coverage:",
                "",
                f"- complete rows: `{values.get('downstream_trajectory_complete_rows', 0)} / {len(rows)}`",
                f"- incomplete rows: `{values.get('downstream_trajectory_incomplete_rows', 0)}`",
                "- Incomplete rows are valid for local replay/risk-head labels, but not for foul/FL/royalty/scoop component-cause analysis.",
                "",
                "Current split:",
                "",
                f"- local EV hard negatives: `{local_count}`",
                f"- whole-game risk only: `{risk_count}`",
                f"- whole-game non-loss controls: `{control_count}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = target_paths(args.input_dir, args.input_jsonl)
    rows, source_rows = collect_rows(paths)
    groups = split_rows(rows)
    summary = summary_rows(rows, source_rows)

    write_jsonl(args.output_dir / "topk_counterfactual_loss_targets_merged.jsonl", rows)
    write_csv(args.output_dir / "topk_counterfactual_loss_targets_merged.csv", rows)
    write_jsonl(args.output_dir / "topk_local_ev_hard_negatives.jsonl", groups["local_ev_hard_negative"])
    write_csv(args.output_dir / "topk_local_ev_hard_negatives.csv", groups["local_ev_hard_negative"])
    write_jsonl(args.output_dir / "topk_whole_game_risk_only.jsonl", groups["whole_game_risk_only"])
    write_csv(args.output_dir / "topk_whole_game_risk_only.csv", groups["whole_game_risk_only"])
    write_jsonl(args.output_dir / "topk_whole_game_non_loss_controls.jsonl", groups["whole_game_non_loss_control"])
    write_csv(args.output_dir / "topk_whole_game_non_loss_controls.csv", groups["whole_game_non_loss_control"])
    write_jsonl(args.output_dir / "topk_requires_local_replay.jsonl", groups["requires_local_replay"])
    write_csv(args.output_dir / "topk_requires_local_replay.csv", groups["requires_local_replay"])
    write_csv(args.output_dir / "topk_loss_target_collection_sources.csv", source_rows)
    write_csv(args.output_dir / "topk_loss_target_collection_downstream_coverage.csv", downstream_coverage_rows(rows))
    write_csv(args.output_dir / "topk_loss_target_collection_summary.csv", summary)
    write_summary_markdown(args.output_dir / "topk_loss_target_collection_summary.md", rows, summary)
    downstream_complete_rows = sum(downstream_complete(row) for row in rows)
    downstream_incomplete_rows = len(rows) - downstream_complete_rows
    manifest = {
        "input_paths": [str(path) for path in paths],
        "output_dir": str(args.output_dir),
        "input_rows": sum(safe_int(row.get("rows")) for row in source_rows),
        "deduped_rows": len(rows),
        "local_ev_hard_negatives": len(groups["local_ev_hard_negative"]),
        "whole_game_risk_only": len(groups["whole_game_risk_only"]),
        "whole_game_non_loss_controls": len(groups["whole_game_non_loss_control"]),
        "requires_local_replay": len(groups["requires_local_replay"]),
        "downstream_trajectory_complete_rows": downstream_complete_rows,
        "downstream_trajectory_incomplete_rows": downstream_incomplete_rows,
        "downstream_trajectory_complete_rate": downstream_complete_rows / len(rows) if rows else 0.0,
    }
    (args.output_dir / "topk_loss_target_collection_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
