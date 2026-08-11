"""Label Stage9f tail-guard high-MC replay results.

The labels produced here are training inputs for a future tail-risk guard.
They are not runtime thresholds and not production evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REPLAY_RESULTS = Path(
    "outputs/evals/hu_turn2_stage9f_tail_guard_replay_tail_loss_mc512/replay_results.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs/training/hu_turn2_stage9f_tail_guard_labels")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-results", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label-name", default="stage9f_tail_guard_labels.csv")
    parser.add_argument("--hard-negative-mean-threshold", type=float, default=-0.25)
    parser.add_argument("--safe-positive-lcb95-threshold", type=float, default=0.0)
    parser.add_argument("--safe-positive-mean-threshold", type=float, default=0.5)
    parser.add_argument("--gray-weight", type=float, default=0.25)
    parser.add_argument("--safe-positive-weight", type=float, default=1.0)
    parser.add_argument("--hard-negative-weight", type=float, default=5.0)
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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def label_for_row(
    row: dict[str, Any],
    *,
    hard_negative_mean_threshold: float,
    safe_positive_lcb95_threshold: float,
    safe_positive_mean_threshold: float,
) -> str:
    gain = safe_float(row.get("gain_mean"))
    lower95 = safe_float(row.get("gain_lower95"))
    if gain <= hard_negative_mean_threshold:
        return "hard_negative"
    if gain >= safe_positive_mean_threshold and lower95 > safe_positive_lcb95_threshold:
        return "safe_positive"
    return "gray"


def training_use(label: str) -> str:
    if label == "hard_negative":
        return "tail_guard_hard_negative"
    if label == "safe_positive":
        return "tail_guard_safe_positive"
    return "tail_guard_gray"


def replay_event_key(row: dict[str, Any]) -> str:
    parts = [
        str(row.get("config_id", "")),
        f"seed{row.get('seed', '')}",
        f"hand{row.get('hand_id', '')}",
        f"base{row.get('baseline_action_index', '')}",
        f"cand{row.get('candidate_action_index', '')}",
    ]
    key = ":".join(parts)
    return key if key.strip(":") else str(row.get("target_id", ""))


def weight_for_label(
    label: str,
    *,
    hard_negative_weight: float,
    safe_positive_weight: float,
    gray_weight: float,
) -> float:
    if label == "hard_negative":
        return hard_negative_weight
    if label == "safe_positive":
        return safe_positive_weight
    return gray_weight


def build_label_row(
    row: dict[str, Any],
    *,
    hard_negative_mean_threshold: float,
    safe_positive_lcb95_threshold: float,
    safe_positive_mean_threshold: float,
    hard_negative_weight: float,
    safe_positive_weight: float,
    gray_weight: float,
    source_path: Path,
) -> dict[str, Any]:
    label = label_for_row(
        row,
        hard_negative_mean_threshold=hard_negative_mean_threshold,
        safe_positive_lcb95_threshold=safe_positive_lcb95_threshold,
        safe_positive_mean_threshold=safe_positive_mean_threshold,
    )
    gain = safe_float(row.get("gain_mean"))
    lower95 = safe_float(row.get("gain_lower95"))
    input_realized = safe_float(row.get("input_realized_delta"))
    return {
        "schema": "hu_turn2_stage9f_tail_guard_label_v1",
        "target_id": row.get("target_id", ""),
        "target_group": row.get("target_group", ""),
        "replay_event_key": replay_event_key(row),
        "config_id": row.get("config_id", ""),
        "seed": row.get("seed", ""),
        "hand_id": row.get("hand_id", ""),
        "seat": row.get("seat", ""),
        "baseline_action_index": row.get("baseline_action_index", ""),
        "candidate_action_index": row.get("candidate_action_index", ""),
        "mc_n": safe_int(row.get("mc_n")),
        "high_mc_gain_mean": gain,
        "high_mc_gain_stderr": safe_float(row.get("gain_stderr")),
        "high_mc_gain_lower90": safe_float(row.get("gain_lower90")),
        "high_mc_gain_lower95": lower95,
        "high_mc_tail_guard_label": label,
        "high_mc_hard_negative_label": int(label == "hard_negative"),
        "high_mc_safe_positive_label": int(label == "safe_positive"),
        "high_mc_gray_label": int(label == "gray"),
        "high_mc_training_weight": weight_for_label(
            label,
            hard_negative_weight=hard_negative_weight,
            safe_positive_weight=safe_positive_weight,
            gray_weight=gray_weight,
        ),
        "recommended_training_use": training_use(label),
        "input_realized_delta": input_realized,
        "input_tail_loss_label": safe_int(row.get("input_tail_loss_label")),
        "input_severe_tail_loss_label": safe_int(row.get("input_severe_tail_loss_label")),
        "input_safe_positive_label": safe_int(row.get("input_safe_positive_label")),
        "sign_flip_vs_input_realized": int((gain > 0.0) != (input_realized > 0.0)),
        "candidate_rank_high_mc": row.get("candidate_rank_high_mc", ""),
        "baseline_rank_high_mc": row.get("baseline_rank_high_mc", ""),
        "paired_delta_le_neg6_rate": safe_float(row.get("paired_delta_le_neg6_rate")),
        "paired_delta_le_neg12_rate": safe_float(row.get("paired_delta_le_neg12_rate")),
        "paired_delta_le_neg20_rate": safe_float(row.get("paired_delta_le_neg20_rate")),
        "source_replay_results": str(source_path),
        "duplicate_source_rows": 1,
        "source_target_groups": row.get("target_group", ""),
        "production_p2_fixed": "No-Go",
        "teacher_50k": "No-Go",
        "t1_training": "No-Go",
    }


def build_labels(
    replay_rows: list[tuple[dict[str, str], Path]],
    *,
    hard_negative_mean_threshold: float,
    safe_positive_lcb95_threshold: float,
    safe_positive_mean_threshold: float,
    hard_negative_weight: float,
    safe_positive_weight: float,
    gray_weight: float,
) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for row, source_path in replay_rows:
        if str(row.get("replay_status", "")) != "success":
            continue
        labels.append(
            build_label_row(
                row,
                hard_negative_mean_threshold=hard_negative_mean_threshold,
                safe_positive_lcb95_threshold=safe_positive_lcb95_threshold,
                safe_positive_mean_threshold=safe_positive_mean_threshold,
                hard_negative_weight=hard_negative_weight,
                safe_positive_weight=safe_positive_weight,
                gray_weight=gray_weight,
                source_path=source_path,
            )
        )
    return labels


def dedupe_label_rows(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    groups_by_key: defaultdict[str, set[str]] = defaultdict(set)
    target_ids_by_key: defaultdict[str, set[str]] = defaultdict(set)
    input_flags = (
        "input_tail_loss_label",
        "input_severe_tail_loss_label",
        "input_safe_positive_label",
    )
    for row in labels:
        key = str(row.get("replay_event_key", "")) or str(row.get("target_id", ""))
        groups_by_key[key].add(str(row.get("target_group", "")))
        target_ids_by_key[key].add(str(row.get("target_id", "")))
        if key not in by_key:
            by_key[key] = dict(row)
            continue
        existing = by_key[key]
        existing["duplicate_source_rows"] = safe_int(existing.get("duplicate_source_rows"), 1) + 1
        for flag in input_flags:
            existing[flag] = max(safe_int(existing.get(flag), 0) or 0, safe_int(row.get(flag), 0) or 0)
    out: list[dict[str, Any]] = []
    for key, row in by_key.items():
        source_groups = sorted(group for group in groups_by_key[key] if group)
        source_target_ids = sorted(target_id for target_id in target_ids_by_key[key] if target_id)
        row["target_group"] = "+".join(source_groups)
        row["source_target_groups"] = ",".join(source_groups)
        row["source_target_ids"] = ",".join(source_target_ids)
        row["deduped_replay_event"] = 1
        out.append(row)
    return out


def summarize(labels: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_label = Counter(str(row.get("high_mc_tail_guard_label", "")) for row in labels)
    by_group_label = Counter(
        (str(row.get("target_group", "")), str(row.get("high_mc_tail_guard_label", ""))) for row in labels
    )
    rows: list[dict[str, Any]] = []
    for (target_group, label), count in sorted(by_group_label.items()):
        group_rows = [
            row
            for row in labels
            if row.get("target_group") == target_group and row.get("high_mc_tail_guard_label") == label
        ]
        gains = [safe_float(row.get("high_mc_gain_mean")) for row in group_rows]
        rows.append(
            {
                "target_group": target_group,
                "label": label,
                "rows": count,
                "gain_mean": sum(gains) / len(gains) if gains else 0.0,
                "gain_min": min(gains, default=0.0),
                "gain_max": max(gains, default=0.0),
                "avg_weight": sum(safe_float(row.get("high_mc_training_weight")) for row in group_rows) / len(group_rows)
                if group_rows
                else 0.0,
            }
        )
    manifest = {
        "schema": "hu_turn2_stage9f_tail_guard_label_manifest_v1",
        "rows": len(labels),
        "label_counts": dict(by_label),
        "hard_negative_mean_threshold": args.hard_negative_mean_threshold,
        "safe_positive_lcb95_threshold": args.safe_positive_lcb95_threshold,
        "safe_positive_mean_threshold": args.safe_positive_mean_threshold,
        "production_p2_fixed": "No-Go",
        "teacher_50k": "No-Go",
        "t1_training": "No-Go",
    }
    return rows, manifest


def write_summary_md(path: Path, *, manifest: dict[str, Any], summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# HU T2 Stage9f Tail-Guard Labels",
        "",
        "These are high-MC replay labels for a future tail-risk guard. They are not",
        "production evidence.",
        "",
        f"- rows: `{manifest['rows']}`",
        f"- label counts: `{manifest['label_counts']}`",
        f"- hard negative threshold: gain <= `{manifest['hard_negative_mean_threshold']}`",
        (
            "- safe positive threshold: gain >= "
            f"`{manifest['safe_positive_mean_threshold']}` and LCB95 > "
            f"`{manifest['safe_positive_lcb95_threshold']}`"
        ),
        "",
        "## Breakdown",
        "",
        "| target group | label | rows | gain mean | min | max | avg weight |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {group} | {label} | {rows} | {mean:+.4f} | {min_gain:+.4f} | {max_gain:+.4f} | {weight:.2f} |".format(
                group=row["target_group"],
                label=row["label"],
                rows=safe_int(row["rows"]),
                mean=safe_float(row["gain_mean"]),
                min_gain=safe_float(row["gain_min"]),
                max_gain=safe_float(row["gain_max"]),
                weight=safe_float(row["avg_weight"]),
            )
        )
    lines.extend(
        [
            "",
            "## Gates",
            "",
            "- production/P2 fixed: `No-Go`",
            "- 50k teacher: `No-Go`",
            "- T1 training: `No-Go`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    replay_paths = args.replay_results or [DEFAULT_REPLAY_RESULTS]
    replay_rows: list[tuple[dict[str, str], Path]] = []
    for path in replay_paths:
        replay_rows.extend((row, path) for row in read_csv(path))
    labels = build_labels(
        replay_rows,
        hard_negative_mean_threshold=args.hard_negative_mean_threshold,
        safe_positive_lcb95_threshold=args.safe_positive_lcb95_threshold,
        safe_positive_mean_threshold=args.safe_positive_mean_threshold,
        hard_negative_weight=args.hard_negative_weight,
        safe_positive_weight=args.safe_positive_weight,
        gray_weight=args.gray_weight,
    )
    summary_rows, manifest = summarize(labels, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    label_csv = args.output_dir / args.label_name
    write_csv(label_csv, labels)
    write_jsonl(args.output_dir / args.label_name.replace(".csv", ".jsonl"), labels)
    dedup_labels = dedupe_label_rows(labels)
    write_csv(args.output_dir / args.label_name.replace(".csv", "_dedup.csv"), dedup_labels)
    write_jsonl(args.output_dir / args.label_name.replace(".csv", "_dedup.jsonl"), dedup_labels)
    write_csv(args.output_dir / "stage9f_tail_guard_label_summary.csv", summary_rows)
    _, dedup_manifest = summarize(dedup_labels, args)
    write_json(
        args.output_dir / "stage9f_tail_guard_label_manifest.json",
        {
            **manifest,
            "label_csv": str(label_csv),
            "dedup_label_csv": str(args.output_dir / args.label_name.replace(".csv", "_dedup.csv")),
            "dedup_rows": len(dedup_labels),
            "dedup_label_counts": dedup_manifest["label_counts"],
            "duplicate_source_rows": len(labels) - len(dedup_labels),
        },
    )
    write_summary_md(args.output_dir / "stage9f_tail_guard_label_summary.md", manifest=manifest, summary_rows=summary_rows)


if __name__ == "__main__":
    main()
