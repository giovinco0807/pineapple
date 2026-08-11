"""Aggregate Stage8c TopK per-fire validation outputs.

This analyzer is intentionally evaluation-only. It combines one or more
``evaluate_hu_turn2_stage8b_topk_mc_rerank`` output directories and reports the
primary metric from realized fired-hand deltas, not from confirm MC gate means.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        action="append",
        type=Path,
        default=[],
        help="Evaluation output directory. Can be repeated.",
    )
    parser.add_argument(
        "--input-glob",
        action="append",
        default=[],
        help="Glob pattern for evaluation output directories. Can be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evals/hu_turn2_current_fl_ev_stage8c_topk_per_fire_aggregate"),
    )
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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil((q / 100.0) * len(ordered)) - 1))
    return ordered[index]


def mean_and_se(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance / len(values))


def confirm_delta(row: dict[str, Any]) -> float:
    value = row.get("confirm_delta")
    if value not in (None, ""):
        return safe_float(value)
    return safe_float(row.get("rerank_delta"))


REPLAY_REQUIRED_FIELDS = (
    "dead_cards",
    "visible_dead_cards",
    "hero_private_discards",
    "opponent_private_discards",
)

MIN_REALIZED_FIRES_FOR_EVIDENCE = 50
PRIMARY_METRIC_SOURCE = "realized_fired_whole_game_delta"
CONFIRM_DELTA_METRIC_ROLE = "gate_diagnostic_only"
PER_FIRE_PERFORMANCE_COLUMN = "per_fire_delta_mean"
HAND_EV_PERFORMANCE_COLUMN = "estimated_ev_per_hand"
CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED = False
NON_FIRED_CANCELLATION_REQUIRED_FOR_PRIMARY_METRIC = True


def missing_replay_fields(row: dict[str, Any]) -> list[str]:
    return [field for field in REPLAY_REQUIRED_FIELDS if not row.get(field)]


def discover_input_dirs(args: argparse.Namespace) -> list[Path]:
    dirs: list[Path] = []
    for directory in args.input_dir or []:
        if not directory.is_dir():
            raise SystemExit(f"--input-dir does not exist or is not a directory: {directory}")
        dirs.append(directory)
    for pattern in args.input_glob or []:
        matches = [path for path in Path().glob(pattern) if path.is_dir()]
        if not matches:
            raise SystemExit(f"--input-glob matched no directories: {pattern}")
        dirs.extend(matches)
    unique: list[Path] = []
    seen = set()
    for path in dirs:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    if not unique:
        raise SystemExit("at least one --input-dir or --input-glob directory is required")
    return unique


def aggregate(input_dirs: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    seed_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    cancellation_rows: list[dict[str, Any]] = []
    for directory in input_dirs:
        for row in read_csv(directory / "seed_breakdown.csv"):
            row["_input_dir"] = str(directory)
            seed_rows.append(row)
        for row in read_jsonl(directory / "runtime_decisions.jsonl"):
            row["_input_dir"] = str(directory)
            decision_rows.append(row)
        for row in read_csv(directory / "cancellation_audit.csv"):
            row["_input_dir"] = str(directory)
            cancellation_rows.append(row)

    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        by_config[str(row.get("config_id", ""))].append(row)

    decisions_by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in decision_rows:
        decisions_by_config[str(row.get("config_id", ""))].append(row)

    cancellation_by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cancellation_rows:
        cancellation_by_config[str(row.get("config_id", ""))].append(row)

    summary_rows: list[dict[str, Any]] = []
    cancellation_summary_rows: list[dict[str, Any]] = []
    for config_id in sorted(set(by_config) | set(decisions_by_config) | set(cancellation_by_config)):
        group = by_config.get(config_id, [])
        paired = sum(safe_int(row.get("paired_seeds")) for row in group)
        weighted_ev_sum = sum(safe_float(row.get("ev_per_hand")) * safe_int(row.get("paired_seeds")) for row in group)
        seed_evs = [safe_float(row.get("ev_per_hand")) for row in group]
        seed_mean, seed_se = mean_and_se(seed_evs)
        weighted_ev = weighted_ev_sum / paired if paired else 0.0

        decisions = decisions_by_config.get(config_id, [])
        fired = [
            row
            for row in decisions
            if row.get("override_fired")
            and row.get("realized_delta_valid")
            and row.get("realized_candidate_seat_delta") not in (None, "")
        ]
        fired_deltas = [safe_float(row.get("realized_candidate_seat_delta")) for row in fired]
        per_fire_mean, per_fire_se = mean_and_se(fired_deltas)
        losses = [max(0.0, -delta) for delta in fired_deltas]
        confirm_deltas = [confirm_delta(row) for row in fired]
        confirm_delta_mean = sum(confirm_deltas) / len(confirm_deltas) if confirm_deltas else 0.0
        negative_predicted = [row for row in fired if safe_float(row.get("predicted_delta")) < 0.0]
        realized_losses = [delta for delta in fired_deltas if delta < 0.0]
        replay_ready = [row for row in decisions if not missing_replay_fields(row)]
        fired_replay_ready = [row for row in fired if not missing_replay_fields(row)]
        replay_missing_counts = Counter(
            field
            for row in decisions
            for field in missing_replay_fields(row)
        )
        fired_replay_missing_counts = Counter(
            field
            for row in fired
            for field in missing_replay_fields(row)
        )
        no_override_counts = Counter(
            str(row.get("no_override_reason", ""))
            for row in decisions
            if not row.get("override_fired")
        )

        decision_count = len(decisions)
        override_count = sum(1 for row in decisions if row.get("override_fired"))
        override_rate = override_count / decision_count if decision_count else 0.0
        summary_rows.append(
            {
                "config_id": config_id,
                "input_dir_count": len({str(row.get("_input_dir", "")) for row in group}) or len(input_dirs),
                "seed_rows": len(group),
                "paired_seeds": paired,
                "weighted_ev_per_hand": weighted_ev,
                "seed_mean_ev_per_hand": seed_mean,
                "seed_ci95_low": seed_mean - 1.96 * seed_se,
                "seed_ci95_high": seed_mean + 1.96 * seed_se,
                "decision_count": decision_count,
                "override_count": override_count,
                "realized_override_count": len(fired_deltas),
                "override_rate": override_rate,
                "decision_replay_ready_count": len(replay_ready),
                "decision_replay_ready_rate": len(replay_ready) / decision_count if decision_count else 0.0,
                "fired_replay_ready_count": len(fired_replay_ready),
                "fired_replay_ready_rate": len(fired_replay_ready) / len(fired) if fired else 0.0,
                "replay_missing_field_counts": json.dumps(dict(replay_missing_counts), sort_keys=True),
                "fired_replay_missing_field_counts": json.dumps(dict(fired_replay_missing_counts), sort_keys=True),
                "per_fire_delta_mean": per_fire_mean,
                "per_fire_delta_ci95_low": per_fire_mean - 1.96 * per_fire_se,
                "per_fire_delta_ci95_high": per_fire_mean + 1.96 * per_fire_se,
                "estimated_ev_per_hand": override_rate * per_fire_mean,
                "estimated_ev_per_hand_ci95_low": override_rate * (per_fire_mean - 1.96 * per_fire_se),
                "estimated_ev_per_hand_ci95_high": override_rate * (per_fire_mean + 1.96 * per_fire_se),
                "confirm_delta_mean_on_fired": confirm_delta_mean,
                "confirm_mean_minus_realized_per_fire": confirm_delta_mean - per_fire_mean,
                "primary_metric_source": PRIMARY_METRIC_SOURCE,
                "per_fire_performance_column": PER_FIRE_PERFORMANCE_COLUMN,
                "hand_ev_performance_column": HAND_EV_PERFORMANCE_COLUMN,
                "confirm_delta_metric_role": CONFIRM_DELTA_METRIC_ROLE,
                "confirm_delta_performance_claim_allowed": CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED,
                "minimum_realized_fires_for_evidence": MIN_REALIZED_FIRES_FOR_EVIDENCE,
                "realized_fire_count_sufficient": len(fired_deltas) >= MIN_REALIZED_FIRES_FOR_EVIDENCE,
                "realized_loss_count": len(realized_losses),
                "p95_loss": percentile(losses, 95),
                "max_loss": max(losses, default=0.0),
                "negative_predicted_delta_override_count": len(negative_predicted),
                "no_override_reason_counts": json.dumps(dict(no_override_counts), sort_keys=True),
            }
        )

        cancel_group = cancellation_by_config.get(config_id, [])
        cancellation_summary_rows.append(
            {
                "config_id": config_id,
                "input_rows": len(cancel_group),
                "valid_realized_delta_count": sum(safe_int(row.get("valid_realized_delta_count")) for row in cancel_group),
                "non_fired_count": sum(safe_int(row.get("non_fired_count")) for row in cancel_group),
                "non_fired_nonzero_count": sum(safe_int(row.get("non_fired_nonzero_count")) for row in cancel_group),
                "non_fired_delta_sum": sum(safe_float(row.get("non_fired_delta_sum")) for row in cancel_group),
                "non_fired_delta_max_abs": max(
                    [safe_float(row.get("non_fired_delta_max_abs")) for row in cancel_group],
                    default=0.0,
                ),
                "fired_count": sum(safe_int(row.get("fired_count")) for row in cancel_group),
                "fired_delta_sum": sum(safe_float(row.get("fired_delta_sum")) for row in cancel_group),
            }
        )
    summary_rows.sort(key=lambda row: safe_float(row.get("estimated_ev_per_hand")), reverse=True)
    return summary_rows, cancellation_summary_rows, seed_rows


def aggregate_position_breakdown(input_dirs: list[Path]) -> list[dict[str, Any]]:
    position_rows: list[dict[str, str]] = []
    decisions_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for directory in input_dirs:
        for row in read_csv(directory / "position_breakdown.csv"):
            position_rows.append(row)
        for row in read_jsonl(directory / "runtime_decisions.jsonl"):
            config_id = str(row.get("config_id", ""))
            seat = str(row.get("seat", ""))
            decisions_by_key[(config_id, seat)].append(row)

    position_by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in position_rows:
        position_by_key[(str(row.get("config_id", "")), str(row.get("seat", "")))].append(row)

    output: list[dict[str, Any]] = []
    for config_id, seat in sorted(set(position_by_key) | set(decisions_by_key)):
        pos_group = position_by_key.get((config_id, seat), [])
        decisions = decisions_by_key.get((config_id, seat), [])
        hands = sum(safe_int(row.get("hands")) for row in pos_group)
        whole_game_ev = (
            sum(safe_float(row.get("ev_per_hand")) * safe_int(row.get("hands")) for row in pos_group) / hands
            if hands
            else 0.0
        )
        fired = [
            row
            for row in decisions
            if row.get("override_fired")
            and row.get("realized_delta_valid")
            and row.get("realized_candidate_seat_delta") not in (None, "")
        ]
        deltas = [safe_float(row.get("realized_candidate_seat_delta")) for row in fired]
        mean, se = mean_and_se(deltas)
        losses = [max(0.0, -delta) for delta in deltas]
        decision_count = len(decisions)
        override_count = sum(1 for row in decisions if row.get("override_fired"))
        override_rate = override_count / decision_count if decision_count else 0.0
        no_override_counts = Counter(
            str(row.get("no_override_reason", ""))
            for row in decisions
            if not row.get("override_fired")
        )
        output.append(
            {
                "config_id": config_id,
                "seat": seat,
                "position_rows": len(pos_group),
                "hands": hands,
                "whole_game_ev_per_hand": whole_game_ev,
                "decision_count": decision_count,
                "override_count": override_count,
                "realized_override_count": len(deltas),
                "override_rate": override_rate,
                "per_fire_delta_mean": mean,
                "per_fire_delta_ci95_low": mean - 1.96 * se,
                "per_fire_delta_ci95_high": mean + 1.96 * se,
                "estimated_ev_per_hand": override_rate * mean,
                "estimated_ev_per_hand_ci95_low": override_rate * (mean - 1.96 * se),
                "estimated_ev_per_hand_ci95_high": override_rate * (mean + 1.96 * se),
                "realized_loss_count": sum(1 for delta in deltas if delta < 0.0),
                "p95_loss": percentile(losses, 95),
                "max_loss": max(losses, default=0.0),
                "no_override_reason_counts": json.dumps(dict(no_override_counts), sort_keys=True),
            }
        )
    return output


def aggregate_risk_veto_candidate_metrics(input_dirs: list[Path]) -> list[dict[str, Any]]:
    by_config: dict[str, list[float]] = defaultdict(list)
    decision_counts: Counter[str] = Counter()
    would_veto_counts: Counter[str] = Counter()
    audit_only_would_veto_counts: Counter[str] = Counter()
    actual_veto_counts: Counter[str] = Counter()

    for directory in input_dirs:
        for row in read_jsonl(directory / "runtime_decisions.jsonl"):
            config_id = str(row.get("config_id", ""))
            decision_counts[config_id] += 1
            if row.get("local_ev_risk_vetoed"):
                actual_veto_counts[config_id] += 1
            if not row.get("local_ev_risk_would_veto"):
                continue
            would_veto_counts[config_id] += 1
            if row.get("local_ev_risk_audit_only"):
                audit_only_would_veto_counts[config_id] += 1
            value = row.get("realized_candidate_seat_delta")
            if value in (None, ""):
                value = row.get("realized_seat_delta")
            if value in (None, ""):
                continue
            delta = safe_float(value, default=float("nan"))
            if math.isfinite(delta):
                by_config[config_id].append(delta)

    output: list[dict[str, Any]] = []
    for config_id in sorted(set(decision_counts) | set(would_veto_counts) | set(by_config)):
        deltas = by_config.get(config_id, [])
        mean, se = mean_and_se(deltas)
        would_veto_count = would_veto_counts[config_id]
        decision_count = decision_counts[config_id]
        would_veto_rate = would_veto_count / decision_count if decision_count else 0.0
        veto_utility = -mean
        losses = [max(0.0, -delta) for delta in deltas]
        output.append(
            {
                "config_id": config_id,
                "decision_count": decision_count,
                "would_veto_count": would_veto_count,
                "audit_only_would_veto_count": audit_only_would_veto_counts[config_id],
                "actual_veto_count": actual_veto_counts[config_id],
                "realized_would_veto_count": len(deltas),
                "would_veto_rate": would_veto_rate,
                "realized_candidate_delta_mean": mean,
                "realized_candidate_delta_std_error": se,
                "realized_candidate_delta_ci95_low": mean - 1.96 * se,
                "realized_candidate_delta_ci95_high": mean + 1.96 * se,
                "veto_utility_per_veto": veto_utility,
                "estimated_veto_utility_per_hand": would_veto_rate * veto_utility,
                "candidate_p95_loss": percentile(losses, 95),
                "candidate_max_loss": max(losses, default=0.0),
            }
        )
    output.sort(key=lambda row: safe_float(row.get("estimated_veto_utility_per_hand")), reverse=True)
    return output


def aggregate_manifest(
    input_dirs: list[Path],
    summary_rows: list[dict[str, Any]],
    cancellation_rows: list[dict[str, Any]],
    risk_veto_rows: list[dict[str, Any]] | None = None,
    position_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    config_ids = {str(row.get("config_id", "")) for row in summary_rows}
    cancellation_config_ids = {str(row.get("config_id", "")) for row in cancellation_rows}
    missing_cancellation_configs = sorted(config_ids - cancellation_config_ids)
    total_non_fired_nonzero = sum(safe_int(row.get("non_fired_nonzero_count")) for row in cancellation_rows)
    total_non_fired_delta_sum = sum(safe_float(row.get("non_fired_delta_sum")) for row in cancellation_rows)
    max_non_fired_delta_abs = max(
        [safe_float(row.get("non_fired_delta_max_abs")) for row in cancellation_rows],
        default=0.0,
    )
    cancellation_audit_present = not missing_cancellation_configs and len(cancellation_rows) >= len(config_ids)
    cancellation_clean = (
        cancellation_audit_present
        and total_non_fired_nonzero == 0
        and abs(total_non_fired_delta_sum) <= 1e-9
        and max_non_fired_delta_abs <= 1e-9
    )
    total_realized_fires = sum(safe_int(row.get("realized_override_count")) for row in summary_rows)
    primary_metric_valid = cancellation_clean and total_realized_fires > 0
    best = summary_rows[0] if summary_rows else {}
    best_realized_fire_count = safe_int(best.get("realized_override_count"))
    best_fire_count_sufficient = best_realized_fire_count >= MIN_REALIZED_FIRES_FOR_EVIDENCE
    best_positive_per_fire_ci = safe_float(best.get("per_fire_delta_ci95_low")) > 0.0
    risk_veto_rows = risk_veto_rows or []
    position_rows = position_rows or []
    best_risk_veto = risk_veto_rows[0] if risk_veto_rows else {}
    risk_veto_realized_count = sum(safe_int(row.get("realized_would_veto_count")) for row in risk_veto_rows)
    position_negative_rows = [
        row
        for row in position_rows
        if safe_int(row.get("realized_override_count")) > 0 and safe_float(row.get("per_fire_delta_ci95_high")) < 0.0
    ]
    position_positive_rows = [
        row
        for row in position_rows
        if safe_int(row.get("realized_override_count")) > 0 and safe_float(row.get("per_fire_delta_ci95_low")) > 0.0
    ]
    manifest = {
        "schema": "hu_turn2_stage8c_topk_per_fire_aggregate_v1",
        "input_dirs": [str(path) for path in input_dirs],
        "primary_metric_source": PRIMARY_METRIC_SOURCE,
        "per_fire_performance_column": PER_FIRE_PERFORMANCE_COLUMN,
        "hand_ev_performance_column": HAND_EV_PERFORMANCE_COLUMN,
        "confirm_delta_metric_role": CONFIRM_DELTA_METRIC_ROLE,
        "confirm_delta_performance_claim_allowed": CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED,
        "non_fired_cancellation_required_for_primary_metric": NON_FIRED_CANCELLATION_REQUIRED_FOR_PRIMARY_METRIC,
        "non_fired_cancellation_requirement": (
            "cancellation_audit must be present for every config, with non_fired_nonzero_count=0, "
            "non_fired_delta_sum=0, and non_fired_delta_max_abs=0"
        ),
        "minimum_realized_fires_for_evidence": MIN_REALIZED_FIRES_FOR_EVIDENCE,
        "config_count": len(summary_rows),
        "cancellation_audit_rows": len(cancellation_rows),
        "missing_cancellation_audit_configs": missing_cancellation_configs,
        "cancellation_audit_present": cancellation_audit_present,
        "cancellation_clean": cancellation_clean,
        "primary_metric_valid": primary_metric_valid,
        "total_paired_seeds": sum(safe_int(row.get("paired_seeds")) for row in summary_rows),
        "total_decisions": sum(safe_int(row.get("decision_count")) for row in summary_rows),
        "total_realized_fires": total_realized_fires,
        "total_non_fired_nonzero_count": total_non_fired_nonzero,
        "total_non_fired_delta_sum": total_non_fired_delta_sum,
        "max_non_fired_delta_abs": max_non_fired_delta_abs,
        "positive_estimated_ev_config_count": sum(
            1 for row in summary_rows if safe_float(row.get("estimated_ev_per_hand")) > 0.0
        ),
        "positive_per_fire_ci_config_count": sum(
            1 for row in summary_rows if safe_float(row.get("per_fire_delta_ci95_low")) > 0.0
        ),
        "best_config_id": best.get("config_id", ""),
        "best_realized_fire_count": best_realized_fire_count,
        "best_realized_fire_count_sufficient": best_fire_count_sufficient,
        "best_estimated_ev_per_hand": safe_float(best.get("estimated_ev_per_hand")),
        "best_per_fire_delta_mean": safe_float(best.get("per_fire_delta_mean")),
        "best_per_fire_delta_ci95_low": safe_float(best.get("per_fire_delta_ci95_low")),
        "best_positive_per_fire_ci": best_positive_per_fire_ci,
        "evidence_decision": (
            "Pass" if primary_metric_valid and best_fire_count_sufficient and best_positive_per_fire_ci else "No-Go"
        ),
        "evaluation_decision": "Pass" if primary_metric_valid else "No-Go",
        "production_p2_fixed": "No-Go",
        "teacher_50k": "No-Go",
        "t1_training": "No-Go",
        "position_breakdown_present": bool(position_rows),
        "position_negative_ci_config_count": len(position_negative_rows),
        "position_positive_ci_config_count": len(position_positive_rows),
    }
    if risk_veto_rows:
        manifest |= {
            "risk_veto_candidate_metric_present": True,
            "risk_veto_candidate_realized_count": risk_veto_realized_count,
            "risk_veto_best_config_id": best_risk_veto.get("config_id", ""),
            "risk_veto_best_utility_per_veto": safe_float(best_risk_veto.get("veto_utility_per_veto")),
            "risk_veto_best_estimated_utility_per_hand": safe_float(
                best_risk_veto.get("estimated_veto_utility_per_hand")
            ),
            "risk_veto_adoption_decision": (
                "Pass"
                if risk_veto_realized_count > 0
                and safe_float(best_risk_veto.get("veto_utility_per_veto")) > 0.0
                and safe_float(best_risk_veto.get("realized_candidate_delta_ci95_high")) < 0.0
                else "No-Go"
            ),
        }
    else:
        manifest |= {
            "risk_veto_candidate_metric_present": False,
            "risk_veto_candidate_realized_count": 0,
            "risk_veto_adoption_decision": "Not-Evaluated",
        }
    return manifest


def write_markdown(
    path: Path,
    summary_rows: list[dict[str, Any]],
    cancellation_rows: list[dict[str, Any]],
    input_dirs: list[Path],
    manifest: dict[str, Any],
    risk_veto_rows: list[dict[str, Any]] | None = None,
    position_rows: list[dict[str, Any]] | None = None,
) -> None:
    lines = [
        "# HU T2 Stage8c TopK Per-Fire Aggregate",
        "",
        "This is validation-only. It does not authorize production, P2 fixed status, T1, or 50k teacher generation.",
        "",
        (
            "Primary performance evidence is the realized whole-game paired delta on fired hands, "
            "with clean non-fired cancellation. Confirm-MC means are gate diagnostics only and must not be used as "
            "performance estimates."
        ),
        "",
        "## Inputs",
        "",
    ]
    for directory in input_dirs:
        lines.append(f"- `{directory}`")
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| config | paired | fires | replay-ready fires | realized-est EV/hand | realized per-fire delta | realized CI low | realized CI high | confirm diag mean | confirm-realized gap | losses | max loss | non-fired nonzero |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    cancellation_by_config = {row["config_id"]: row for row in cancellation_rows}
    for row in summary_rows:
        cancel = cancellation_by_config.get(str(row["config_id"]), {})
        lines.append(
            "| {config} | {paired} | {fires} | {ready_fires} | {ev:.4f} | {mean:.4f} | {low:.4f} | {high:.4f} | {confirm:.4f} | {gap:.4f} | {losses} | {max_loss:.4f} | {nonzero} |".format(
                config=row["config_id"],
                paired=row["paired_seeds"],
                fires=row["realized_override_count"],
                ready_fires=row["fired_replay_ready_count"],
                ev=safe_float(row["estimated_ev_per_hand"]),
                mean=safe_float(row["per_fire_delta_mean"]),
                low=safe_float(row["per_fire_delta_ci95_low"]),
                high=safe_float(row["per_fire_delta_ci95_high"]),
                confirm=safe_float(row["confirm_delta_mean_on_fired"]),
                gap=safe_float(row["confirm_mean_minus_realized_per_fire"]),
                losses=row["realized_loss_count"],
                max_loss=safe_float(row["max_loss"]),
                nonzero=cancel.get("non_fired_nonzero_count", 0),
            )
        )
    position_rows = position_rows or []
    if position_rows:
        lines.extend(
            [
                "",
                "## Position Breakdown",
                "",
                "| config | seat | hands | whole-game EV/hand | fires | per-fire delta | CI low | CI high | estimated EV/hand |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in position_rows:
            lines.append(
                "| {config} | {seat} | {hands} | {whole_ev:.4f} | {fires} | {mean:.4f} | {low:.4f} | {high:.4f} | {ev:.4f} |".format(
                    config=row["config_id"],
                    seat=row["seat"],
                    hands=row["hands"],
                    whole_ev=safe_float(row["whole_game_ev_per_hand"]),
                    fires=row["realized_override_count"],
                    mean=safe_float(row["per_fire_delta_mean"]),
                    low=safe_float(row["per_fire_delta_ci95_low"]),
                    high=safe_float(row["per_fire_delta_ci95_high"]),
                    ev=safe_float(row["estimated_ev_per_hand"]),
                )
            )
    risk_veto_rows = risk_veto_rows or []
    if risk_veto_rows:
        lines.extend(
            [
                "",
                "## Risk Veto Candidate Metrics",
                "",
                "Audit-only rows estimate whether a veto would have helped. Positive candidate delta means the veto would have blocked a profitable candidate.",
                "",
                "| config | would veto | realized | candidate delta | CI low | CI high | veto utility | utility/hand | actual veto |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in risk_veto_rows:
            lines.append(
                "| {config} | {would} | {realized} | {mean:.4f} | {low:.4f} | {high:.4f} | {utility:.4f} | {ev:.6f} | {actual} |".format(
                    config=row["config_id"],
                    would=row["would_veto_count"],
                    realized=row["realized_would_veto_count"],
                    mean=safe_float(row["realized_candidate_delta_mean"]),
                    low=safe_float(row["realized_candidate_delta_ci95_low"]),
                    high=safe_float(row["realized_candidate_delta_ci95_high"]),
                    utility=safe_float(row["veto_utility_per_veto"]),
                    ev=safe_float(row["estimated_veto_utility_per_hand"]),
                    actual=row["actual_veto_count"],
                )
            )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- primary metric valid: `{manifest['primary_metric_valid']}`",
            f"- primary metric source: `{manifest['primary_metric_source']}`",
            f"- per-fire performance column: `{manifest['per_fire_performance_column']}`",
            f"- hand EV performance column: `{manifest['hand_ev_performance_column']}`",
            f"- confirm delta role: `{manifest['confirm_delta_metric_role']}`",
            f"- confirm delta performance claim allowed: `{manifest['confirm_delta_performance_claim_allowed']}`",
            f"- non-fired cancellation required: `{manifest['non_fired_cancellation_required_for_primary_metric']}`",
            f"- cancellation clean: `{manifest['cancellation_clean']}`",
            f"- evidence decision: `{manifest['evidence_decision']}`",
            f"- minimum realized fires for evidence: `{manifest['minimum_realized_fires_for_evidence']}`",
            f"- position breakdown present: `{manifest['position_breakdown_present']}`",
            f"- risk veto adoption: `{manifest['risk_veto_adoption_decision']}`",
            "- execution: `Pass`",
            "- production / P2 fixed: `No-Go`",
            "- 50k teacher: `No-Go`",
            "- T1 training: `No-Go`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dirs = discover_input_dirs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows, cancellation_rows, seed_rows = aggregate(input_dirs)
    risk_veto_rows = aggregate_risk_veto_candidate_metrics(input_dirs)
    position_rows = aggregate_position_breakdown(input_dirs)
    manifest = aggregate_manifest(input_dirs, summary_rows, cancellation_rows, risk_veto_rows, position_rows)
    write_csv(args.output_dir / "aggregate_summary.csv", summary_rows)
    write_csv(args.output_dir / "aggregate_cancellation_audit.csv", cancellation_rows)
    write_csv(args.output_dir / "aggregate_seed_breakdown.csv", seed_rows)
    write_csv(args.output_dir / "aggregate_position_breakdown.csv", position_rows)
    write_csv(args.output_dir / "aggregate_risk_veto_candidate_metrics.csv", risk_veto_rows)
    (args.output_dir / "aggregate_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(
        args.output_dir / "aggregate_summary.md",
        summary_rows,
        cancellation_rows,
        input_dirs,
        manifest,
        risk_veto_rows,
        position_rows,
    )
    print(
        json.dumps(
            manifest | {"output_dir": str(args.output_dir)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
