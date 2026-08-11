"""Prepare Stage9g tail-risk guard rows from Stage9f high-MC labels.

This joins replay targets, which carry the board/action/runtime features, with
deduped high-MC labels, which carry the safer train/eval target.  The output is
compatible with ``train_hu_turn2_stage8c_risk_head`` using
``--target-mode whole_game_risk``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_TARGETS = Path(
    "outputs/training/hu_turn2_stage9f_cse2_csemax2_tail_guard_targets_10000/stage9f_tail_guard_targets.jsonl"
)
DEFAULT_LABELS = Path(
    "outputs/evals/hu_turn2_stage9f_tail_guard_replay_all_mc512_gcp/labels/stage9f_tail_guard_labels_dedup.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs/training/hu_turn2_stage9g_tail_guard_training")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--include-gray", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


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


def target_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("target_id", "")): row for row in rows if row.get("target_id")}


def source_target_ids(label: dict[str, Any]) -> list[str]:
    text = str(label.get("source_target_ids", "") or label.get("target_id", ""))
    return [part.strip() for part in text.split(",") if part.strip()]


def recommended_training_use(label: str) -> str:
    if label == "hard_negative":
        return "whole_game_risk_only"
    if label == "safe_positive":
        return "whole_game_non_loss_control"
    return "stage9g_tail_guard_gray"


def build_training_row(label: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    high_mc_label = str(label.get("high_mc_tail_guard_label", ""))
    gain = safe_float(label.get("high_mc_gain_mean"))
    stderr = safe_float(label.get("high_mc_gain_stderr"))
    use = recommended_training_use(high_mc_label)
    baseline_index = safe_int(target.get("baseline_action_index"), -1)
    candidate_index = safe_int(target.get("candidate_action_index"), -1)
    row = dict(target)
    row.update(
        {
            "schema": "hu_turn2_stage9g_tail_guard_training_row_v1",
            "stage9g_source": "stage9f_tail_guard_high_mc_dedup",
            "stage9g_label_target_id": label.get("target_id", ""),
            "stage9g_replay_event_key": label.get("replay_event_key", ""),
            "stage9g_high_mc_label": high_mc_label,
            "stage9g_high_mc_hard_negative_label": safe_int(label.get("high_mc_hard_negative_label")),
            "stage9g_high_mc_safe_positive_label": safe_int(label.get("high_mc_safe_positive_label")),
            "stage9g_high_mc_gray_label": safe_int(label.get("high_mc_gray_label")),
            "stage9g_high_mc_gain_mean": gain,
            "stage9g_high_mc_gain_stderr": stderr,
            "stage9g_high_mc_gain_lower95": safe_float(label.get("high_mc_gain_lower95")),
            "stage9g_high_mc_training_weight": safe_float(label.get("high_mc_training_weight"), 1.0),
            "stage9g_duplicate_source_rows": safe_int(label.get("duplicate_source_rows"), 1),
            "stage9g_source_target_groups": label.get("source_target_groups", ""),
            "stage9g_source_target_ids": label.get("source_target_ids", ""),
            "recommended_training_use": use,
            "use_for_whole_game_risk_head": int(use in {"whole_game_risk_only", "whole_game_non_loss_control"}),
            "realized_delta": gain,
            "realized_delta_observed": 1,
            "baseline_index": baseline_index,
            "candidate_index": candidate_index,
            "local_replay_status": "ok",
            "local_replay_action_mapping_status": "ok",
            "local_replay_delta": gain,
            "local_replay_delta_se": stderr,
            "local_replay_future_samples": safe_int(label.get("mc_n"), safe_int(target.get("mc_n"), 512)),
            "local_replay_label": "negative" if high_mc_label == "hard_negative" else high_mc_label,
            "local_replay_bucket": f"stage9g_{high_mc_label}",
            "candidate_source": "stage9f_tail_guard_high_mc",
            "candidate_index_source": "candidate_action_index",
            "production_p2_fixed": "No-Go",
            "teacher_50k": "No-Go",
            "t1_training": "No-Go",
        }
    )
    return row


def build_rows(
    targets: list[dict[str, Any]],
    labels: list[dict[str, str]],
    *,
    include_gray: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    targets_by_id = target_index(targets)
    output: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for label in labels:
        high_mc_label = str(label.get("high_mc_tail_guard_label", ""))
        if high_mc_label == "gray" and not include_gray:
            continue
        target = None
        for target_id in source_target_ids(label):
            target = targets_by_id.get(target_id)
            if target is not None:
                break
        if target is None:
            missing.append({"label_target_id": label.get("target_id", ""), "source_target_ids": label.get("source_target_ids", "")})
            continue
        output.append(build_training_row(label, target))
    return output, missing


def write_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# HU T2 Stage9g Tail-Guard Training Rows",
        "",
        "These rows merge Stage9f high-MC tail-guard labels back onto replay targets.",
        "They are for guard training only, not production evidence.",
        "",
        f"- rows: `{manifest['rows']}`",
        f"- label counts: `{manifest['label_counts']}`",
        f"- missing labels: `{manifest['missing_labels']}`",
        f"- include gray: `{manifest['include_gray']}`",
        "",
        "## Gates",
        "",
        "- production/P2 fixed: `No-Go`",
        "- 50k teacher: `No-Go`",
        "- T1 training: `No-Go`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    targets = read_jsonl(args.targets)
    labels = read_csv(args.labels)
    rows, missing = build_rows(targets, labels, include_gray=args.include_gray)
    label_counts = Counter(str(row.get("stage9g_high_mc_label", "")) for row in rows)
    use_counts = Counter(str(row.get("recommended_training_use", "")) for row in rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "stage9g_tail_guard_training_rows.jsonl"
    write_jsonl(rows_path, rows)
    write_csv(args.output_dir / "stage9g_tail_guard_training_rows.csv", rows)
    write_json(args.output_dir / "stage9g_tail_guard_missing_labels.json", {"missing": missing})
    manifest = {
        "schema": "hu_turn2_stage9g_tail_guard_training_manifest_v1",
        "targets_path": str(args.targets),
        "labels_path": str(args.labels),
        "training_rows_jsonl": str(rows_path),
        "rows": len(rows),
        "missing_labels": len(missing),
        "label_counts": dict(label_counts),
        "recommended_training_use_counts": dict(use_counts),
        "include_gray": bool(args.include_gray),
        "production_p2_fixed": "No-Go",
        "teacher_50k": "No-Go",
        "t1_training": "No-Go",
    }
    write_json(args.output_dir / "stage9g_tail_guard_training_manifest.json", manifest)
    write_summary(args.output_dir / "stage9g_tail_guard_training_summary.md", manifest)


if __name__ == "__main__":
    main()
