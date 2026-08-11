"""Audit realized tail losses from HU T2 Stage9f TopK+confirm runs.

The primary evidence remains realized whole-game paired delta on fired hands.
Confirm-MC deltas are only used here as diagnostics for noisy gates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REPLAY_REQUIRED_FIELDS = (
    "dead_cards",
    "visible_dead_cards",
    "hero_private_discards",
    "opponent_private_discards",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", type=Path, default=[])
    parser.add_argument("--input-glob", action="append", default=[])
    parser.add_argument("--config-id", action="append", default=[])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evals/hu_turn2_stage9f_tail_loss_audit"),
    )
    parser.add_argument("--top-n", type=int, default=50)
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil((q / 100.0) * len(ordered)) - 1))
    return ordered[index]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def decision_log_path(directory: Path) -> Path:
    runtime_path = directory / "runtime_decisions.jsonl"
    if runtime_path.exists():
        return runtime_path
    return directory / "topk_decisions.jsonl"


def row_config_id(row: dict[str, Any]) -> str:
    config_id = str(row.get("config_id") or "")
    if config_id:
        return config_id
    profile = str(row.get("runtime_profile") or "")
    return profile or "unknown"


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


def discover_input_dirs(args: argparse.Namespace) -> list[Path]:
    dirs: list[Path] = []
    for directory in args.input_dir:
        if not directory.is_dir():
            raise SystemExit(f"--input-dir does not exist or is not a directory: {directory}")
        dirs.append(directory)
    for pattern in args.input_glob:
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


def missing_replay_fields(row: dict[str, Any]) -> list[str]:
    return [field for field in REPLAY_REQUIRED_FIELDS if not row.get(field)]


def is_fired_realized(row: dict[str, Any]) -> bool:
    return (
        bool(row.get("override_fired"))
        and bool(row.get("realized_delta_valid"))
        and row.get("realized_candidate_seat_delta") not in (None, "")
    )


def confirm_delta(row: dict[str, Any]) -> float:
    if row.get("confirm_delta") not in (None, ""):
        return safe_float(row.get("confirm_delta"))
    return safe_float(row.get("rerank_delta"))


def loss_components(row: dict[str, Any]) -> list[str]:
    components: list[str] = []
    if safe_float(row.get("fl_delta_vs_baseline")) < -1e-9:
        components.append("lost_fl_value")
    if safe_float(row.get("foul_delta_vs_baseline")) < -1e-9:
        components.append("foul_regression")
    if safe_float(row.get("royalty_delta_vs_baseline")) < -1e-9:
        components.append("royalty_regression")
    if safe_float(row.get("line_score_delta_vs_baseline")) < -1e-9:
        components.append("line_regression")
    if safe_float(row.get("scoop_delta_vs_baseline")) < -1e-9:
        components.append("scoop_regression")
    if row.get("candidate_downstream_override_fired") or row.get("baseline_downstream_override_fired"):
        components.append("downstream_override_changed")
    if safe_float(row.get("confirm_delta_se")) >= 2.0:
        components.append("high_confirm_se")
    if safe_int(row.get("candidate_ev_rank"), default=99) > 1:
        components.append("non_top_rank_candidate")
    return components or ["unclassified"]


def primary_loss_label(row: dict[str, Any]) -> str:
    components = set(loss_components(row))
    priority = [
        "lost_fl_value",
        "foul_regression",
        "royalty_regression",
        "line_regression",
        "scoop_regression",
        "downstream_override_changed",
        "high_confirm_se",
        "non_top_rank_candidate",
    ]
    for label in priority:
        if label in components:
            return label
    return "unclassified"


def compact_action(action: Any) -> str:
    if not isinstance(action, dict):
        return ""
    placements = action.get("placements") or []
    discards = action.get("discards") or []
    return json.dumps({"placements": placements, "discards": discards}, sort_keys=True)


def audit_rows(input_dirs: list[Path], config_filter: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for directory in input_dirs:
        source_log = decision_log_path(directory)
        for row in read_jsonl(source_log):
            config_id = row_config_id(row)
            if config_filter and config_id not in config_filter:
                continue
            if not is_fired_realized(row):
                continue
            delta = safe_float(row.get("realized_candidate_seat_delta"))
            if delta >= 0.0:
                continue
            components = loss_components(row)
            rows.append(
                {
                    "config_id": config_id,
                    "seed": row.get("seed"),
                    "hand_id": row.get("hand_id"),
                    "seat": row.get("seat"),
                    "seat_swap": row.get("seat_swap"),
                    "realized_delta": delta,
                    "loss": -delta,
                    "primary_loss_label": primary_loss_label(row),
                    "loss_components": ";".join(components),
                    "confirm_delta": confirm_delta(row),
                    "confirm_delta_se": safe_float(row.get("confirm_delta_se")),
                    "confirm_z": confirm_delta(row) / safe_float(row.get("confirm_delta_se"), 1.0)
                    if safe_float(row.get("confirm_delta_se")) > 0.0
                    else 0.0,
                    "stage_a_delta": safe_float(row.get("stage_a_delta")),
                    "stage_a_delta_se": safe_float(row.get("stage_a_delta_se")),
                    "predicted_delta": safe_float(row.get("predicted_delta")),
                    "gate_probability": safe_float(row.get("gate_probability")),
                    "candidate_ev_rank": safe_int(row.get("candidate_ev_rank"), default=-1),
                    "royalty_delta_vs_baseline": safe_float(row.get("royalty_delta_vs_baseline")),
                    "fl_delta_vs_baseline": safe_float(row.get("fl_delta_vs_baseline")),
                    "line_score_delta_vs_baseline": safe_float(row.get("line_score_delta_vs_baseline")),
                    "scoop_delta_vs_baseline": safe_float(row.get("scoop_delta_vs_baseline")),
                    "foul_delta_vs_baseline": safe_float(row.get("foul_delta_vs_baseline")),
                    "hero_royalty_vs_baseline": safe_float(row.get("hero_royalty_vs_baseline")),
                    "opponent_royalty_vs_baseline": safe_float(row.get("opponent_royalty_vs_baseline")),
                    "hero_fl_value_vs_baseline": safe_float(row.get("hero_fl_value_vs_baseline")),
                    "opponent_fl_value_vs_baseline": safe_float(row.get("opponent_fl_value_vs_baseline")),
                    "candidate_downstream_override_fired": bool(row.get("candidate_downstream_override_fired")),
                    "baseline_downstream_override_fired": bool(row.get("baseline_downstream_override_fired")),
                    "replay_ready": not missing_replay_fields(row),
                    "missing_replay_fields": ";".join(missing_replay_fields(row)),
                    "hand_seed": row.get("hand_seed", row.get("seed")),
                    "game_id": row.get("game_id", row.get("hand_id")),
                    "source_log": str(source_log),
                    "hero_board": row.get("hero_board") or {},
                    "opponent_board": row.get("opponent_board") or {},
                    "dead_cards": row.get("dead_cards") or [],
                    "visible_dead_cards": row.get("visible_dead_cards") or [],
                    "hero_private_discards": row.get("hero_private_discards") or [],
                    "opponent_private_discards": row.get("opponent_private_discards") or [],
                    "cards_to_place": row.get("cards_to_place") or [],
                    "baseline_action": row.get("baseline_action") or {},
                    "candidate_action": row.get("final_action") or {},
                    "final_action": row.get("final_action") or {},
                    "baseline_action_index": row.get("baseline_action_index"),
                    "candidate_action_index": row.get("final_action_index"),
                    "hero_board_json": json.dumps(row.get("hero_board") or {}, sort_keys=True),
                    "opponent_board_json": json.dumps(row.get("opponent_board") or {}, sort_keys=True),
                    "dead_cards_json": json.dumps(row.get("dead_cards") or [], sort_keys=True),
                    "cards_to_place_json": json.dumps(row.get("cards_to_place") or [], sort_keys=True),
                    "baseline_action_json": compact_action(row.get("baseline_action")),
                    "final_action_json": compact_action(row.get("final_action")),
                    "_input_dir": str(directory),
                }
            )
    rows.sort(key=lambda item: safe_float(item.get("loss")), reverse=True)
    return rows


def fired_decisions(input_dirs: list[Path], config_filter: set[str]) -> tuple[list[dict[str, Any]], Counter[str], Counter[str]]:
    fired: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    first_decision_counts: Counter[str] = Counter()
    for directory in input_dirs:
        for row in read_jsonl(decision_log_path(directory)):
            config_id = row_config_id(row)
            if config_filter and config_id not in config_filter:
                continue
            decision_counts[config_id] += 1
            if row.get("seat") == "first":
                first_decision_counts[config_id] += 1
            if is_fired_realized(row):
                fired_row = dict(row)
                fired_row["config_id"] = config_id
                fired.append(fired_row)
    return fired, decision_counts, first_decision_counts


def mean_and_ci(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, mean, mean
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    se = math.sqrt(variance / len(values))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def guard_definitions() -> list[tuple[str, Any]]:
    return [
        ("all", lambda row: True),
        ("rank<=1", lambda row: safe_int(row.get("candidate_ev_rank"), default=99) <= 1),
        ("rank<=2", lambda row: safe_int(row.get("candidate_ev_rank"), default=99) <= 2),
        ("rank<=3", lambda row: safe_int(row.get("candidate_ev_rank"), default=99) <= 3),
        ("confirm_se<=2.0", lambda row: safe_float(row.get("confirm_delta_se")) <= 2.0),
        ("confirm_se<=2.5", lambda row: safe_float(row.get("confirm_delta_se")) <= 2.5),
        ("confirm_se<=3.0", lambda row: safe_float(row.get("confirm_delta_se")) <= 3.0),
        ("confirm_z>=1.75", lambda row: confirm_delta(row) >= 1.75 * safe_float(row.get("confirm_delta_se"))),
        ("confirm_z>=2.0", lambda row: confirm_delta(row) >= 2.0 * safe_float(row.get("confirm_delta_se"))),
        ("confirm_z>=2.5", lambda row: confirm_delta(row) >= 2.5 * safe_float(row.get("confirm_delta_se"))),
        ("predicted_delta>=0.25", lambda row: safe_float(row.get("predicted_delta")) >= 0.25),
        ("predicted_delta>=0.5", lambda row: safe_float(row.get("predicted_delta")) >= 0.5),
        ("predicted_delta>=1.0", lambda row: safe_float(row.get("predicted_delta")) >= 1.0),
        ("predicted_delta>=1.5", lambda row: safe_float(row.get("predicted_delta")) >= 1.5),
        (
            "rank<=1 & confirm_se<=2.5",
            lambda row: safe_int(row.get("candidate_ev_rank"), default=99) <= 1
            and safe_float(row.get("confirm_delta_se")) <= 2.5,
        ),
        (
            "rank<=1 & confirm_z>=1.75",
            lambda row: safe_int(row.get("candidate_ev_rank"), default=99) <= 1
            and confirm_delta(row) >= 1.75 * safe_float(row.get("confirm_delta_se")),
        ),
        (
            "confirm_se<=2.5 & confirm_z>=1.75",
            lambda row: safe_float(row.get("confirm_delta_se")) <= 2.5
            and confirm_delta(row) >= 1.75 * safe_float(row.get("confirm_delta_se")),
        ),
    ]


def guard_sweep_rows(
    fired: list[dict[str, Any]],
    decision_counts: Counter[str],
    first_decision_counts: Counter[str],
) -> list[dict[str, Any]]:
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fired:
        by_config[str(row.get("config_id", ""))].append(row)

    output: list[dict[str, Any]] = []
    for config_id, group in sorted(by_config.items()):
        for guard_id, predicate in guard_definitions():
            selected = [row for row in group if predicate(row)]
            deltas = [safe_float(row.get("realized_candidate_seat_delta")) for row in selected]
            mean, low, high = mean_and_ci(deltas)
            losses = [max(0.0, -delta) for delta in deltas]
            all_denominator = decision_counts[config_id]
            first_denominator = first_decision_counts[config_id]
            output.append(
                {
                    "config_id": config_id,
                    "guard_id": guard_id,
                    "fired_count": len(selected),
                    "all_decision_count": all_denominator,
                    "first_decision_count": first_denominator,
                    "fire_rate_all_decisions": len(selected) / all_denominator if all_denominator else 0.0,
                    "fire_rate_first_decisions": len(selected) / first_denominator if first_denominator else 0.0,
                    "per_fire_delta_mean": mean,
                    "per_fire_delta_ci95_low": low,
                    "per_fire_delta_ci95_high": high,
                    "estimated_ev_per_all_decision": (len(selected) / all_denominator * mean) if all_denominator else 0.0,
                    "estimated_ev_per_first_decision": (len(selected) / first_denominator * mean) if first_denominator else 0.0,
                    "loss_count": sum(1 for delta in deltas if delta < 0.0),
                    "p95_loss": percentile(losses, 95),
                    "max_loss": max(losses, default=0.0),
                }
            )
    output.sort(
        key=lambda row: (
            str(row.get("config_id", "")),
            -safe_float(row.get("estimated_ev_per_all_decision")),
            safe_float(row.get("max_loss")),
        )
    )
    return output


def summarize(losses: list[dict[str, Any]], input_dirs: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in losses:
        by_config[str(row.get("config_id", ""))].append(row)

    summary_rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    for config_id, group in sorted(by_config.items()):
        loss_values = [safe_float(row.get("loss")) for row in group]
        replay_ready_count = sum(1 for row in group if row.get("replay_ready"))
        summary_rows.append(
            {
                "config_id": config_id,
                "input_dir_count": len(input_dirs),
                "loss_count": len(group),
                "replay_ready_loss_count": replay_ready_count,
                "replay_ready_loss_rate": replay_ready_count / len(group) if group else 0.0,
                "avg_loss": sum(loss_values) / len(loss_values) if loss_values else 0.0,
                "p50_loss": percentile(loss_values, 50),
                "p90_loss": percentile(loss_values, 90),
                "p95_loss": percentile(loss_values, 95),
                "p99_loss": percentile(loss_values, 99),
                "max_loss": max(loss_values, default=0.0),
                "lost_fl_value_count": sum("lost_fl_value" in str(row.get("loss_components", "")) for row in group),
                "foul_regression_count": sum("foul_regression" in str(row.get("loss_components", "")) for row in group),
                "royalty_regression_count": sum("royalty_regression" in str(row.get("loss_components", "")) for row in group),
                "line_regression_count": sum("line_regression" in str(row.get("loss_components", "")) for row in group),
                "high_confirm_se_count": sum("high_confirm_se" in str(row.get("loss_components", "")) for row in group),
            }
        )
        primary_counts = Counter(str(row.get("primary_loss_label", "")) for row in group)
        component_counts: Counter[str] = Counter()
        for row in group:
            component_counts.update(str(row.get("loss_components", "")).split(";"))
        for label, count in sorted(primary_counts.items()):
            breakdown_rows.append(
                {
                    "config_id": config_id,
                    "breakdown_type": "primary",
                    "label": label,
                    "count": count,
                    "rate": count / len(group) if group else 0.0,
                }
            )
        for label, count in sorted(component_counts.items()):
            breakdown_rows.append(
                {
                    "config_id": config_id,
                    "breakdown_type": "component",
                    "label": label,
                    "count": count,
                    "rate": count / len(group) if group else 0.0,
                }
            )
    return summary_rows, breakdown_rows


def write_summary_md(
    path: Path,
    summary_rows: list[dict[str, Any]],
    breakdown_rows: list[dict[str, Any]],
    top_rows: list[dict[str, Any]],
    guard_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# HU T2 Stage9f Tail-Loss Audit",
        "",
        "This audit uses realized fired whole-game paired deltas only. Confirm-MC",
        "deltas are diagnostic and are not treated as performance evidence.",
        "",
        "## Summary",
        "",
        "| config | losses | replay-ready | avg loss | p95 loss | max loss | lost FL | foul | royalty | high confirm SE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {config} | {losses} | {ready} | {avg:.4f} | {p95:.4f} | {max_loss:.4f} | {fl} | {foul} | {royalty} | {se} |".format(
                config=row["config_id"],
                losses=safe_int(row["loss_count"]),
                ready=safe_int(row["replay_ready_loss_count"]),
                avg=safe_float(row["avg_loss"]),
                p95=safe_float(row["p95_loss"]),
                max_loss=safe_float(row["max_loss"]),
                fl=safe_int(row["lost_fl_value_count"]),
                foul=safe_int(row["foul_regression_count"]),
                royalty=safe_int(row["royalty_regression_count"]),
                se=safe_int(row["high_confirm_se_count"]),
            )
        )
    lines.extend(["", "## Primary Loss Labels", ""])
    for row in breakdown_rows:
        if row.get("breakdown_type") != "primary":
            continue
        lines.append(
            "- `{config}` `{label}`: `{count}` ({rate:.1%})".format(
                config=row["config_id"],
                label=row["label"],
                count=safe_int(row["count"]),
                rate=safe_float(row["rate"]),
            )
        )
    lines.extend(["", "## Worst Losses", ""])
    for row in top_rows[:10]:
        lines.append(
            "- `{config}` seed `{seed}` hand `{hand}` loss `{loss:.4f}` label `{label}` confirm `{confirm:.4f}` se `{se:.4f}`".format(
                config=row["config_id"],
                seed=row["seed"],
                hand=row["hand_id"],
                loss=safe_float(row["loss"]),
                label=row["primary_loss_label"],
                confirm=safe_float(row["confirm_delta"]),
                se=safe_float(row["confirm_delta_se"]),
            )
        )
    lines.extend(["", "## Guard Sweep Highlights", ""])
    for row in guard_rows[:8]:
        lines.append(
            "- `{guard}` fires `{fires}`, EV/all `{ev_all:+.4f}`, EV/first `{ev_first:+.4f}`, per-fire `{per:+.4f}`, max loss `{max_loss:.4f}`".format(
                guard=row["guard_id"],
                fires=safe_int(row["fired_count"]),
                ev_all=safe_float(row["estimated_ev_per_all_decision"]),
                ev_first=safe_float(row["estimated_ev_per_first_decision"]),
                per=safe_float(row["per_fire_delta_mean"]),
                max_loss=safe_float(row["max_loss"]),
            )
        )
    lines.extend(
        [
            "",
            "## Decision Use",
            "",
            "- Use this audit to decide whether a veto/guard is needed before any first-seat-only runtime packaging.",
            "- Do not use confirm delta averages as adoption evidence.",
            "- If replay-ready loss count is complete, the worst losses can be fed into selected high-MC replay or hard-negative training.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dirs = discover_input_dirs(args)
    losses = audit_rows(input_dirs, set(args.config_id))
    fired, decision_counts, first_decision_counts = fired_decisions(input_dirs, set(args.config_id))
    guard_rows = guard_sweep_rows(fired, decision_counts, first_decision_counts)
    summary_rows, breakdown_rows = summarize(losses, input_dirs)
    top_rows = losses[: max(0, args.top_n)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "tail_loss_summary.csv", summary_rows)
    write_csv(args.output_dir / "tail_loss_component_breakdown.csv", breakdown_rows)
    write_csv(args.output_dir / "tail_loss_guard_sweep.csv", guard_rows)
    write_csv(args.output_dir / "tail_loss_top.csv", top_rows)
    write_jsonl(args.output_dir / "tail_loss_top.jsonl", top_rows)
    write_jsonl(
        args.output_dir / "tail_loss_replay_candidates.jsonl",
        [row for row in top_rows if row.get("replay_ready")],
    )
    write_summary_md(args.output_dir / "tail_loss_audit.md", summary_rows, breakdown_rows, top_rows, guard_rows)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "input_dir_count": len(input_dirs),
                "loss_count": len(losses),
                "summary_rows": len(summary_rows),
                "top_rows": len(top_rows),
                "guard_rows": len(guard_rows),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
