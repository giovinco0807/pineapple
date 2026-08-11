"""Prepare Stage9f fired rows for tail-risk guard training or high-MC replay.

The input is one or more Stage9f profile-canary/evaluate_matchups output
directories containing a ``topk_decisions.jsonl`` or ``runtime_decisions.jsonl``.
Only fired rows with realized whole-game paired deltas are selected.  The output
is not performance evidence; it is a replay-ready target set for the next guard
or selected high-MC pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT_DIR = Path("outputs/training/hu_turn2_stage9f_tail_guard_targets")
REPLAY_REQUIRED_FIELDS = (
    "hero_board",
    "opponent_board",
    "cards_to_place",
    "baseline_action",
    "final_action",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", type=Path, default=[])
    parser.add_argument("--input-glob", action="append", default=[])
    parser.add_argument("--config-id", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-name", default="stage9f_tail_guard_targets.jsonl")
    parser.add_argument("--loss-limit", type=int, default=200)
    parser.add_argument("--positive-limit", type=int, default=200)
    parser.add_argument("--zero-limit", type=int, default=100)
    parser.add_argument("--boundary-limit", type=int, default=200)
    parser.add_argument("--severe-loss-threshold", type=float, default=8.0)
    parser.add_argument("--safe-positive-threshold", type=float, default=2.0)
    parser.add_argument("--confirm-z-threshold", type=float, default=2.0)
    parser.add_argument("--confirm-z-boundary-width", type=float, default=0.35)
    parser.add_argument(
        "--dedupe-event-key",
        action="store_true",
        help="Replay each config/hand/baseline/candidate event once even if it appears in multiple target groups.",
    )
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


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


def decision_log_path(directory: Path) -> Path:
    runtime_path = directory / "runtime_decisions.jsonl"
    if runtime_path.exists():
        return runtime_path
    return directory / "topk_decisions.jsonl"


def discover_input_dirs(input_dirs: list[Path], input_globs: list[str]) -> list[Path]:
    dirs: list[Path] = []
    for directory in input_dirs:
        if not directory.is_dir():
            raise SystemExit(f"--input-dir does not exist or is not a directory: {directory}")
        dirs.append(directory)
    for pattern in input_globs:
        matches = [path for path in Path().glob(pattern) if path.is_dir()]
        if not matches:
            raise SystemExit(f"--input-glob matched no directories: {pattern}")
        dirs.extend(matches)
    unique: list[Path] = []
    seen: set[Path] = set()
    for directory in dirs:
        resolved = directory.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(directory)
    if not unique:
        raise SystemExit("at least one --input-dir or --input-glob is required")
    return unique


def row_config_id(row: dict[str, Any]) -> str:
    return str(row.get("config_id") or row.get("runtime_profile") or "unknown")


def realized_delta(row: dict[str, Any]) -> float:
    if row.get("realized_candidate_seat_delta") not in (None, ""):
        return safe_float(row.get("realized_candidate_seat_delta"))
    return safe_float(row.get("realized_delta"))


def confirm_delta(row: dict[str, Any]) -> float:
    if row.get("confirm_delta") not in (None, ""):
        return safe_float(row.get("confirm_delta"))
    return safe_float(row.get("rerank_delta"))


def confirm_z(row: dict[str, Any]) -> float:
    se = safe_float(row.get("confirm_delta_se"))
    return confirm_delta(row) / se if se > 0.0 else 0.0


def is_fired_realized(row: dict[str, Any]) -> bool:
    return (
        bool(row.get("override_fired"))
        and bool(row.get("realized_delta_valid"))
        and row.get("realized_candidate_seat_delta") not in (None, "")
    )


def missing_replay_fields(row: dict[str, Any]) -> list[str]:
    missing = [field for field in REPLAY_REQUIRED_FIELDS if not row.get(field)]
    if not row.get("hero_private_discards") and not row.get("visible_dead_cards"):
        missing.append("hero_visible_discard")
    return missing


def target_id(row: dict[str, Any], group: str) -> str:
    config_id = row_config_id(row)
    seed = row.get("seed", row.get("hand_seed", "na"))
    hand_id = row.get("hand_id", row.get("game_id", "na"))
    candidate_index = row.get("final_action_index", row.get("candidate_action_index", "na"))
    return f"{config_id}:{group}:seed{seed}:hand{hand_id}:cand{candidate_index}"


def replay_event_key(row: dict[str, Any]) -> str:
    config_id = str(row.get("config_id") or row_config_id(row))
    seed = str(row.get("seed", row.get("hand_seed", "na")))
    hand_id = str(row.get("hand_id", row.get("game_id", "na")))
    baseline_action = str(row.get("baseline_action_json") or compact_action(row.get("baseline_action")))
    candidate_action = str(row.get("candidate_action_json") or compact_action(row.get("candidate_action")))
    return "|".join((config_id, seed, hand_id, baseline_action, candidate_action))


def compact_action(action: Any) -> str:
    if not isinstance(action, dict):
        return ""
    return json.dumps(
        {
            "placements": action.get("placements") or [],
            "discards": action.get("discards") or [],
        },
        sort_keys=True,
    )


def build_target(
    row: dict[str, Any],
    *,
    target_group: str,
    severe_loss_threshold: float,
    safe_positive_threshold: float,
    source_log: Path,
) -> dict[str, Any]:
    delta = realized_delta(row)
    loss = max(0.0, -delta)
    missing = missing_replay_fields(row)
    out = {
        "schema": "hu_turn2_stage9f_tail_guard_target_v1",
        "target_id": target_id(row, target_group),
        "target_group": target_group,
        "config_id": row_config_id(row),
        "seed": row.get("seed"),
        "hand_seed": row.get("hand_seed", row.get("seed")),
        "hand_id": row.get("hand_id", row.get("game_id")),
        "game_id": row.get("game_id", row.get("hand_id")),
        "seat": row.get("seat"),
        "seat_swap": row.get("seat_swap"),
        "realized_delta": delta,
        "loss": loss,
        "tail_loss_label": int(delta < 0.0),
        "severe_tail_loss_label": int(loss >= severe_loss_threshold),
        "safe_positive_label": int(delta >= safe_positive_threshold),
        "zero_realized_label": int(abs(delta) <= 1e-12),
        "confirm_delta": confirm_delta(row),
        "confirm_delta_se": safe_float(row.get("confirm_delta_se")),
        "confirm_z": confirm_z(row),
        "stage_a_delta": safe_float(row.get("stage_a_delta")),
        "stage_a_delta_se": safe_float(row.get("stage_a_delta_se")),
        "predicted_delta": safe_float(row.get("predicted_delta")),
        "gate_probability": safe_float(row.get("gate_probability")),
        "candidate_ev_rank": safe_int(row.get("candidate_ev_rank"), 99),
        "baseline_action_index": row.get("baseline_action_index"),
        "candidate_action_index": row.get("final_action_index", row.get("candidate_action_index")),
        "hero_board": row.get("hero_board") or {},
        "opponent_board": row.get("opponent_board") or {},
        "dead_cards": row.get("dead_cards") or [],
        "visible_dead_cards": row.get("visible_dead_cards") or [],
        "hero_private_discards": row.get("hero_private_discards") or [],
        "opponent_private_discards": row.get("opponent_private_discards") or [],
        "cards_to_place": row.get("cards_to_place") or [],
        "baseline_action": row.get("baseline_action") or {},
        "candidate_action": row.get("final_action") or row.get("candidate_action") or {},
        "baseline_action_json": compact_action(row.get("baseline_action")),
        "candidate_action_json": compact_action(row.get("final_action") or row.get("candidate_action")),
        "replay_ready": not missing,
        "missing_replay_fields": ",".join(missing),
        "source_log": str(source_log),
        "replay_event_key": "",
        "source_target_groups": target_group,
        "duplicate_source_rows": 1,
        "observed_performance_claim": "No",
        "production_p2_fixed": "No-Go",
        "teacher_50k": "No-Go",
        "t1_training": "No-Go",
        "note": "Selected from realized fired rows for guard training/high-MC replay; not production evidence.",
    }
    out["replay_event_key"] = replay_event_key(out)
    return out


def dedupe_targets(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("target_id", ""))
        previous = best.get(key)
        if previous is None or safe_float(row.get("loss")) > safe_float(previous.get("loss")):
            best[key] = row
    return list(best.values())


def target_group_priority(group: str) -> int:
    return {
        "tail_loss": 0,
        "confirm_z_boundary": 1,
        "positive_control": 2,
        "zero_control": 3,
    }.get(group, 99)


def dedupe_targets_by_replay_event(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    groups_by_key: dict[str, set[str]] = {}
    duplicate_counts: Counter[str] = Counter()
    for row in rows:
        key = str(row.get("replay_event_key") or replay_event_key(row))
        groups_by_key.setdefault(key, set()).add(str(row.get("target_group") or ""))
        duplicate_counts[key] += 1
        previous = best.get(key)
        if previous is None:
            best[key] = row
            continue
        current_priority = target_group_priority(str(row.get("target_group") or ""))
        previous_priority = target_group_priority(str(previous.get("target_group") or ""))
        if (
            current_priority < previous_priority
            or (
                current_priority == previous_priority
                and safe_float(row.get("loss")) > safe_float(previous.get("loss"))
            )
        ):
            best[key] = row
    deduped: list[dict[str, Any]] = []
    for key, row in best.items():
        out = dict(row)
        out["source_target_groups"] = ",".join(sorted(group for group in groups_by_key.get(key, set()) if group))
        out["duplicate_source_rows"] = duplicate_counts[key]
        deduped.append(out)
    return deduped


def select_targets(
    fired_rows: list[tuple[dict[str, Any], Path]],
    *,
    loss_limit: int,
    positive_limit: int,
    zero_limit: int,
    boundary_limit: int,
    severe_loss_threshold: float,
    safe_positive_threshold: float,
    confirm_z_threshold: float,
    confirm_z_boundary_width: float,
    dedupe_event_key: bool = False,
) -> list[dict[str, Any]]:
    losses = [(row, path) for row, path in fired_rows if realized_delta(row) < 0.0]
    positives = [(row, path) for row, path in fired_rows if realized_delta(row) >= safe_positive_threshold]
    zeros = [(row, path) for row, path in fired_rows if abs(realized_delta(row)) <= 1e-12]
    boundary = [
        (row, path)
        for row, path in fired_rows
        if abs(confirm_z(row) - confirm_z_threshold) <= confirm_z_boundary_width
    ]

    targets: list[dict[str, Any]] = []
    for row, path in sorted(losses, key=lambda item: (-max(0.0, -realized_delta(item[0])), str(item[1])))[:loss_limit]:
        targets.append(
            build_target(
                row,
                target_group="tail_loss",
                severe_loss_threshold=severe_loss_threshold,
                safe_positive_threshold=safe_positive_threshold,
                source_log=path,
            )
        )
    for row, path in sorted(positives, key=lambda item: (-realized_delta(item[0]), str(item[1])))[:positive_limit]:
        targets.append(
            build_target(
                row,
                target_group="positive_control",
                severe_loss_threshold=severe_loss_threshold,
                safe_positive_threshold=safe_positive_threshold,
                source_log=path,
            )
        )
    for row, path in sorted(zeros, key=lambda item: (abs(confirm_z(item[0]) - confirm_z_threshold), str(item[1])))[:zero_limit]:
        targets.append(
            build_target(
                row,
                target_group="zero_control",
                severe_loss_threshold=severe_loss_threshold,
                safe_positive_threshold=safe_positive_threshold,
                source_log=path,
            )
        )
    for row, path in sorted(
        boundary,
        key=lambda item: (abs(confirm_z(item[0]) - confirm_z_threshold), -abs(realized_delta(item[0])), str(item[1])),
    )[:boundary_limit]:
        targets.append(
            build_target(
                row,
                target_group="confirm_z_boundary",
                severe_loss_threshold=severe_loss_threshold,
                safe_positive_threshold=safe_positive_threshold,
                source_log=path,
            )
        )
    targets = dedupe_targets(targets)
    if dedupe_event_key:
        targets = dedupe_targets_by_replay_event(targets)
    return sorted(targets, key=lambda row: (str(row.get("target_group")), -safe_float(row.get("loss")), str(row.get("target_id"))))


def collect_fired_rows(input_dirs: list[Path], config_filter: set[str]) -> tuple[list[tuple[dict[str, Any], Path]], Counter[str]]:
    rows: list[tuple[dict[str, Any], Path]] = []
    counters: Counter[str] = Counter()
    for directory in input_dirs:
        source_log = decision_log_path(directory)
        if not source_log.exists():
            counters["missing_source_log"] += 1
            continue
        for row in read_jsonl(source_log):
            config_id = row_config_id(row)
            if config_filter and config_id not in config_filter:
                counters["filtered_config"] += 1
                continue
            counters["rows_seen"] += 1
            if not bool(row.get("override_fired")):
                counters["not_fired"] += 1
                continue
            if not is_fired_realized(row):
                counters["fired_missing_realized_delta"] += 1
                continue
            counters["fired_realized"] += 1
            rows.append((row, source_log))
    return rows, counters


def summarize_targets(targets: list[dict[str, Any]], counters: Counter[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    group_counts = Counter(str(row.get("target_group", "")) for row in targets)
    label_counts = Counter(
        (
            str(row.get("target_group", "")),
            safe_int(row.get("tail_loss_label")),
            safe_int(row.get("severe_tail_loss_label")),
            safe_int(row.get("safe_positive_label")),
        )
        for row in targets
    )
    summary_rows: list[dict[str, Any]] = []
    for group, count in sorted(group_counts.items()):
        group_rows = [row for row in targets if row.get("target_group") == group]
        losses = [safe_float(row.get("loss")) for row in group_rows]
        deltas = [safe_float(row.get("realized_delta")) for row in group_rows]
        summary_rows.append(
            {
                "target_group": group,
                "rows": count,
                "replay_ready": sum(1 for row in group_rows if row.get("replay_ready")),
                "tail_losses": sum(safe_int(row.get("tail_loss_label")) for row in group_rows),
                "severe_tail_losses": sum(safe_int(row.get("severe_tail_loss_label")) for row in group_rows),
                "safe_positives": sum(safe_int(row.get("safe_positive_label")) for row in group_rows),
                "delta_mean": sum(deltas) / len(deltas) if deltas else 0.0,
                "loss_max": max(losses, default=0.0),
                "rank_gt1": sum(1 for row in group_rows if safe_int(row.get("candidate_ev_rank"), 99) > 1),
            }
        )
    manifest = {
        "schema": "hu_turn2_stage9f_tail_guard_target_manifest_v1",
        "targets": len(targets),
        "replay_ready": sum(1 for row in targets if row.get("replay_ready")),
        "counters": dict(counters),
        "target_group_counts": dict(group_counts),
        "label_counts": {
            f"{group}|tail{tail}|severe{severe}|safe{safe}": count
            for (group, tail, severe, safe), count in label_counts.items()
        },
        "production_p2_fixed": "No-Go",
        "teacher_50k": "No-Go",
        "t1_training": "No-Go",
    }
    return summary_rows, manifest


def write_summary_md(path: Path, *, manifest: dict[str, Any], summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# HU T2 Stage9f Tail-Guard Targets",
        "",
        "These rows are selected from realized fired Stage9f runtime logs for",
        "tail-risk guard training or selected high-MC replay. They are not",
        "production evidence.",
        "",
        f"- targets: `{manifest['targets']}`",
        f"- replay-ready: `{manifest['replay_ready']} / {manifest['targets']}`",
        f"- source fired realized rows: `{manifest['counters'].get('fired_realized', 0)}`",
        f"- production/P2 fixed: `{manifest['production_p2_fixed']}`",
        f"- 50k teacher: `{manifest['teacher_50k']}`",
        f"- T1 training: `{manifest['t1_training']}`",
        "",
        "## Breakdown",
        "",
        "| group | rows | replay-ready | losses | severe losses | safe positives | delta mean | max loss | rank>1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {group} | {rows} | {ready} | {losses} | {severe} | {safe} | {delta:+.4f} | {max_loss:.4f} | {rank_gt1} |".format(
                group=row["target_group"],
                rows=safe_int(row["rows"]),
                ready=safe_int(row["replay_ready"]),
                losses=safe_int(row["tail_losses"]),
                severe=safe_int(row["severe_tail_losses"]),
                safe=safe_int(row["safe_positives"]),
                delta=safe_float(row["delta_mean"]),
                max_loss=safe_float(row["loss_max"]),
                rank_gt1=safe_int(row["rank_gt1"]),
            )
        )
    lines.extend(
        [
            "",
            "## Next",
            "",
            "- Use this as the input set for a tail-risk guard or selected high-MC replay.",
            "- Keep `stage9f_cse2_csemax2_firstseat` validation-only until the tail guard",
            "  is independently validated.",
            "- Do not start T1, 50k teacher, production, or P2 fixed from this artifact.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dirs = discover_input_dirs(args.input_dir, args.input_glob)
    config_filter = {str(value) for value in args.config_id if str(value)}
    fired_rows, counters = collect_fired_rows(input_dirs, config_filter)
    targets = select_targets(
        fired_rows,
        loss_limit=args.loss_limit,
        positive_limit=args.positive_limit,
        zero_limit=args.zero_limit,
        boundary_limit=args.boundary_limit,
        severe_loss_threshold=args.severe_loss_threshold,
        safe_positive_threshold=args.safe_positive_threshold,
        confirm_z_threshold=args.confirm_z_threshold,
        confirm_z_boundary_width=args.confirm_z_boundary_width,
        dedupe_event_key=args.dedupe_event_key,
    )
    summary_rows, manifest = summarize_targets(targets, counters)
    manifest.update(
        {
            "input_dirs": [str(path) for path in input_dirs],
            "config_filter": sorted(config_filter),
            "target_jsonl": str(args.output_dir / args.target_name),
            "target_csv": str(args.output_dir / args.target_name.replace(".jsonl", ".csv")),
            "loss_limit": args.loss_limit,
            "positive_limit": args.positive_limit,
            "zero_limit": args.zero_limit,
            "boundary_limit": args.boundary_limit,
            "severe_loss_threshold": args.severe_loss_threshold,
            "safe_positive_threshold": args.safe_positive_threshold,
            "confirm_z_threshold": args.confirm_z_threshold,
            "confirm_z_boundary_width": args.confirm_z_boundary_width,
            "dedupe_event_key": args.dedupe_event_key,
            "duplicate_source_rows": sum(max(0, safe_int(row.get("duplicate_source_rows"), 1) - 1) for row in targets),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / args.target_name, targets)
    write_csv(args.output_dir / args.target_name.replace(".jsonl", ".csv"), targets)
    write_csv(args.output_dir / "stage9f_tail_guard_target_summary.csv", summary_rows)
    write_json(args.output_dir / "stage9f_tail_guard_target_manifest.json", manifest)
    write_summary_md(args.output_dir / "stage9f_tail_guard_target_summary.md", manifest=manifest, summary_rows=summary_rows)


if __name__ == "__main__":
    main()
