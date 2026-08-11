"""Summarize Stage9f profile canary TopK decision logs.

This analyzer is for runtime plumbing and safety checks. When profile logs
include realized fired whole-game deltas, those deltas are reported separately
from confirm-MC diagnostics.
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
    "baseline_action",
    "final_action",
)

CONFIRM_DELTA_METRIC_ROLE = "gate_diagnostic_only"
CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--matchup-summary", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evals/hu_turn2_stage9f_profile_canary_audit"),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


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


def safe_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil((q / 100.0) * len(ordered)) - 1))
    return ordered[index]


def missing_replay_fields(row: dict[str, Any]) -> list[str]:
    return [field for field in REPLAY_REQUIRED_FIELDS if not row.get(field)]


def latency_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    values = [safe_float(row.get("runtime_latency_ms")) for row in rows]
    values = [value for value in values if value > 0.0]
    if not values:
        return {
            "latency_ms_mean": 0.0,
            "latency_ms_p50": 0.0,
            "latency_ms_p90": 0.0,
            "latency_ms_p95": 0.0,
            "latency_ms_max": 0.0,
        }
    return {
        "latency_ms_mean": sum(values) / len(values),
        "latency_ms_p50": percentile(values, 50),
        "latency_ms_p90": percentile(values, 90),
        "latency_ms_p95": percentile(values, 95),
        "latency_ms_max": max(values),
    }


def latency_component_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    runtime_values: list[float] = []
    stage_a_values: list[float] = []
    confirm_values: list[float] = []
    overhead_values: list[float] = []
    for row in rows:
        runtime = safe_float(row.get("runtime_latency_ms"))
        stage_a = safe_float(row.get("mc_rerank_latency_ms"))
        confirm = safe_float(row.get("confirm_mc_latency_ms"))
        if runtime <= 0.0:
            continue
        runtime_values.append(runtime)
        stage_a_values.append(stage_a)
        confirm_values.append(confirm)
        overhead_values.append(max(0.0, runtime - stage_a - confirm))

    if not runtime_values:
        return {
            "latency_component_runtime_p95_ms": 0.0,
            "latency_component_stage_a_p95_ms": 0.0,
            "latency_component_confirm_p95_ms": 0.0,
            "latency_component_overhead_p95_ms": 0.0,
            "latency_component_stage_a_mean_ms": 0.0,
            "latency_component_confirm_mean_ms": 0.0,
            "latency_component_overhead_mean_ms": 0.0,
        }

    return {
        "latency_component_runtime_p95_ms": percentile(runtime_values, 95),
        "latency_component_stage_a_p95_ms": percentile(stage_a_values, 95),
        "latency_component_confirm_p95_ms": percentile(confirm_values, 95),
        "latency_component_overhead_p95_ms": percentile(overhead_values, 95),
        "latency_component_stage_a_mean_ms": sum(stage_a_values) / len(stage_a_values),
        "latency_component_confirm_mean_ms": sum(confirm_values) / len(confirm_values),
        "latency_component_overhead_mean_ms": sum(overhead_values) / len(overhead_values),
    }


def reason_bucket(row: dict[str, Any]) -> str:
    if bool(row.get("override_fired")):
        return "override_fired"
    return str(row.get("no_override_reason") or "fallback_to_baseline")


def realized_delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("realized_delta_valid")
        and row.get("realized_candidate_seat_delta") not in (None, "")
    ]


def realized_delta_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    valid = realized_delta_rows(rows)
    fired = [row for row in valid if bool(row.get("override_fired"))]
    non_fired = [row for row in valid if not row.get("override_fired")]
    fired_deltas = [safe_float(row.get("realized_candidate_seat_delta")) for row in fired]
    non_fired_deltas = [safe_float(row.get("realized_candidate_seat_delta")) for row in non_fired]
    nonzero_non_fired = [delta for delta in non_fired_deltas if abs(delta) > 1e-9]
    per_fire = sum(fired_deltas) / len(fired_deltas) if fired_deltas else 0.0
    if len(fired_deltas) >= 2:
        variance = sum((delta - per_fire) ** 2 for delta in fired_deltas) / (len(fired_deltas) - 1)
        per_fire_se = math.sqrt(variance / len(fired_deltas))
    else:
        per_fire_se = 0.0
    losses = [max(0.0, -delta) for delta in fired_deltas]
    decision_count = len(rows)
    override_rate = len(fired) / decision_count if decision_count else 0.0
    return {
        "realized_delta_count": len(valid),
        "realized_override_count": len(fired),
        "realized_per_fire_delta_mean": per_fire,
        "realized_per_fire_delta_se": per_fire_se,
        "realized_per_fire_delta_ci95_low": per_fire - 1.96 * per_fire_se,
        "realized_per_fire_delta_ci95_high": per_fire + 1.96 * per_fire_se,
        "realized_per_fire_delta_min": min(fired_deltas, default=0.0),
        "realized_per_fire_delta_max": max(fired_deltas, default=0.0),
        "realized_per_fire_positive_count": sum(1 for delta in fired_deltas if delta > 1e-9),
        "realized_per_fire_negative_count": sum(1 for delta in fired_deltas if delta < -1e-9),
        "realized_per_fire_zero_count": sum(1 for delta in fired_deltas if abs(delta) <= 1e-9),
        "realized_per_fire_loss_p90": percentile(losses, 90),
        "realized_per_fire_loss_p95": percentile(losses, 95),
        "realized_per_fire_loss_p99": percentile(losses, 99),
        "realized_per_fire_loss_max": max(losses, default=0.0),
        "realized_estimated_ev_per_decision": override_rate * per_fire,
        "non_fired_count_with_realized_delta": len(non_fired),
        "non_fired_nonzero_count": len(nonzero_non_fired),
        "non_fired_delta_sum": sum(non_fired_deltas),
        "non_fired_delta_max_abs": max((abs(delta) for delta in non_fired_deltas), default=0.0),
    }


def fired_loss_rows(rows: list[dict[str, Any]], *, limit: int = 30) -> list[dict[str, Any]]:
    fired = [
        row
        for row in realized_delta_rows(rows)
        if bool(row.get("override_fired"))
    ]
    ordered = sorted(
        fired,
        key=lambda row: safe_float(row.get("realized_candidate_seat_delta")),
    )
    output: list[dict[str, Any]] = []
    for row in ordered[:limit]:
        delta = safe_float(row.get("realized_candidate_seat_delta"))
        output.append(
            {
                "loss": max(0.0, -delta),
                "realized_candidate_seat_delta": delta,
                "paired_index": row.get("paired_index"),
                "hand_seed": row.get("hand_seed"),
                "game_id": row.get("game_id"),
                "hand_id": row.get("hand_id"),
                "seat_swap": row.get("seat_swap"),
                "seat": row.get("seat"),
                "hero_board": row.get("hero_board"),
                "opponent_board": row.get("opponent_board"),
                "cards_to_place": row.get("cards_to_place"),
                "dead_cards": row.get("dead_cards"),
                "visible_dead_cards": row.get("visible_dead_cards"),
                "hero_private_discards": row.get("hero_private_discards"),
                "opponent_private_discards": row.get("opponent_private_discards"),
                "baseline_action": row.get("baseline_action"),
                "final_action": row.get("final_action"),
                "baseline_action_index": row.get("baseline_action_index"),
                "final_action_index": row.get("final_action_index"),
                "candidate_ev_rank": row.get("candidate_ev_rank"),
                "predicted_delta": row.get("predicted_delta"),
                "gate_probability": row.get("gate_probability"),
                "model_score": row.get("model_score"),
                "stage_a_delta": row.get("stage_a_delta"),
                "stage_a_delta_se": row.get("stage_a_delta_se"),
                "confirm_delta": row.get("confirm_delta"),
                "confirm_delta_se": row.get("confirm_delta_se"),
                "confirm_delta_count": row.get("confirm_delta_count"),
                "runtime_latency_ms": row.get("runtime_latency_ms"),
                "runtime_profile": row.get("runtime_profile"),
                "t3_continuation_policy": row.get("t3_continuation_policy"),
            }
        )
    return output


def summarize_decisions(
    rows: list[dict[str, Any]],
    *,
    decisions_path: Path,
    matchup_summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    total = len(rows)
    overrides = [row for row in rows if bool(row.get("override_fired"))]
    first_rows = [row for row in rows if row.get("seat") == "first"]
    second_rows = [row for row in rows if row.get("seat") == "second"]
    confirm_rows = [row for row in rows if row.get("confirm_delta") not in (None, "")]
    stage_a_rows = [row for row in rows if row.get("stage_a_delta") not in (None, "")]
    missing_counts: Counter[str] = Counter()
    replay_ready = 0
    for row in rows:
        missing = missing_replay_fields(row)
        if not missing:
            replay_ready += 1
        missing_counts.update(missing)

    summary = [
        {
            "decisions_path": str(decisions_path),
            "matchup_summary_path": str(matchup_summary.get("_path", "")),
            "profile_a": matchup_summary.get("profile_a", ""),
            "profile_b": matchup_summary.get("profile_b", ""),
            "paired_seeds": matchup_summary.get("paired_seeds", ""),
            "hands": matchup_summary.get("hands", ""),
            "avg_score_per_hand_for_a": matchup_summary.get("avg_score_per_hand_for_a", ""),
            "decision_count": total,
            "override_count": len(overrides),
            "override_rate": len(overrides) / total if total else 0.0,
            "first_decision_count": len(first_rows),
            "first_override_count": sum(1 for row in first_rows if bool(row.get("override_fired"))),
            "second_decision_count": len(second_rows),
            "second_override_count": sum(1 for row in second_rows if bool(row.get("override_fired"))),
            "confirm_evaluated_count": len(confirm_rows),
            "confirm_delta_metric_role": CONFIRM_DELTA_METRIC_ROLE,
            "confirm_delta_performance_claim_allowed": CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED,
            "stage_a_evaluated_count": len(stage_a_rows),
            "replay_ready_count": replay_ready,
            "replay_ready_rate": replay_ready / total if total else 0.0,
            "missing_replay_field_counts": json.dumps(dict(sorted(missing_counts.items())), sort_keys=True),
            **realized_delta_summary(rows),
            **latency_summary(rows),
            **latency_component_summary(rows),
        }
    ]

    reason_counter = Counter(reason_bucket(row) for row in rows)
    reason_rows = [
        {
            "bucket_type": "no_override_reason",
            "bucket": reason,
            "decision_count": count,
            "override_count": sum(
                1
                for row in rows
                if reason_bucket(row) == reason and bool(row.get("override_fired"))
            ),
            "share": count / total if total else 0.0,
        }
        for reason, count in sorted(reason_counter.items())
    ]

    seat_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        seat_groups[str(row.get("seat") or "")].append(row)
    seat_rows = []
    for seat, group in sorted(seat_groups.items()):
        group_overrides = [row for row in group if bool(row.get("override_fired"))]
        seat_rows.append(
            {
                "seat": seat,
                "decision_count": len(group),
                "override_count": len(group_overrides),
                "override_rate": len(group_overrides) / len(group) if group else 0.0,
                "confirm_evaluated_count": sum(1 for row in group if row.get("confirm_delta") not in (None, "")),
                "replay_ready_count": sum(1 for row in group if not missing_replay_fields(row)),
                **realized_delta_summary(group),
                **latency_summary(group),
                **latency_component_summary(group),
            }
        )

    profile_counter = Counter(str(row.get("runtime_profile") or "") for row in rows)
    policy_counter = Counter(str(row.get("t3_continuation_policy") or "") for row in rows)
    config_rows = [
        {
            "bucket_type": "runtime_profile",
            "bucket": key,
            "decision_count": value,
        }
        for key, value in sorted(profile_counter.items())
    ] + [
        {
            "bucket_type": "t3_continuation_policy",
            "bucket": key,
            "decision_count": value,
        }
        for key, value in sorted(policy_counter.items())
    ]
    return summary, reason_rows, seat_rows, config_rows


def write_summary_md(
    path: Path,
    summary: dict[str, Any],
    reason_rows: list[dict[str, Any]],
    seat_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# HU T2 Stage9f Profile Canary Audit",
        "",
        "This is a runtime-log audit. Realized fired whole-game deltas are used",
        "only when the profile log contains `realized_candidate_seat_delta`.",
        "",
        "## Summary",
        "",
        f"- profile A: `{summary.get('profile_a', '')}`",
        f"- profile B: `{summary.get('profile_b', '')}`",
        f"- paired seeds: `{summary.get('paired_seeds', '')}`",
        f"- hands: `{summary.get('hands', '')}`",
        f"- decisions: `{summary.get('decision_count', 0)}`",
        f"- overrides: `{summary.get('override_count', 0)}`",
        f"- override rate: `{float(summary.get('override_rate', 0.0)):.4f}`",
        f"- realized overrides: `{summary.get('realized_override_count', 0)}`",
        f"- confirm delta role: `{summary.get('confirm_delta_metric_role', CONFIRM_DELTA_METRIC_ROLE)}`",
        f"- confirm delta performance claim allowed: "
        f"`{summary.get('confirm_delta_performance_claim_allowed', CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED)}`",
        f"- realized per-fire delta: `{float(summary.get('realized_per_fire_delta_mean', 0.0)):.4f}`",
        f"- realized per-fire 95% CI: "
        f"`[{float(summary.get('realized_per_fire_delta_ci95_low', 0.0)):.4f}, "
        f"{float(summary.get('realized_per_fire_delta_ci95_high', 0.0)):.4f}]`",
        f"- realized per-fire negative count: `{summary.get('realized_per_fire_negative_count', 0)}`",
        f"- realized per-fire p95/max loss: "
        f"`{float(summary.get('realized_per_fire_loss_p95', 0.0)):.4f}` / "
        f"`{float(summary.get('realized_per_fire_loss_max', 0.0)):.4f}`",
        f"- estimated EV/decision: `{float(summary.get('realized_estimated_ev_per_decision', 0.0)):.4f}`",
        f"- non-fired nonzero realized deltas: `{summary.get('non_fired_nonzero_count', 0)}`",
        f"- replay-ready: `{summary.get('replay_ready_count', 0)} / {summary.get('decision_count', 0)}`",
        f"- latency p95 ms: `{float(summary.get('latency_ms_p95', 0.0)):.2f}`",
        f"- latency p95 components ms: "
        f"stage A `{float(summary.get('latency_component_stage_a_p95_ms', 0.0)):.2f}`, "
        f"confirm `{float(summary.get('latency_component_confirm_p95_ms', 0.0)):.2f}`, "
        f"overhead `{float(summary.get('latency_component_overhead_p95_ms', 0.0)):.2f}`",
        "",
        "## No-Override Reasons",
        "",
        "| reason | decisions | share |",
        "|---|---:|---:|",
    ]
    for row in reason_rows:
        lines.append(
            f"| `{row['bucket']}` | {row['decision_count']} | {float(row['share']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Seat Breakdown",
            "",
            "| seat | decisions | overrides | realized | per-fire | non-fired nonzero | confirm eval | replay-ready | p95 ms | stage A p95 | confirm p95 | overhead p95 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in seat_rows:
        lines.append(
            "| {seat} | {decision_count} | {override_count} | {realized_override_count} | "
            "{realized_per_fire_delta_mean:.4f} | {non_fired_nonzero_count} | "
            "{confirm_evaluated_count} | {replay_ready_count} | {latency_ms_p95:.2f} | "
            "{latency_component_stage_a_p95_ms:.2f} | {latency_component_confirm_p95_ms:.2f} | "
            "{latency_component_overhead_p95_ms:.2f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.decisions)
    matchup_summary = read_json(args.matchup_summary)
    if args.matchup_summary is not None:
        matchup_summary["_path"] = str(args.matchup_summary)
    summary_rows, reason_rows, seat_rows, config_rows = summarize_decisions(
        rows,
        decisions_path=args.decisions,
        matchup_summary=matchup_summary,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "profile_canary_decision_summary.csv", summary_rows)
    write_csv(args.output_dir / "profile_canary_reason_breakdown.csv", reason_rows)
    write_csv(args.output_dir / "profile_canary_seat_breakdown.csv", seat_rows)
    write_csv(args.output_dir / "profile_canary_config_breakdown.csv", config_rows)
    write_jsonl(
        args.output_dir / "profile_canary_fired_losses_top30.jsonl",
        fired_loss_rows(rows),
    )
    write_summary_md(
        args.output_dir / "profile_canary_summary.md",
        summary_rows[0] if summary_rows else {},
        reason_rows,
        seat_rows,
    )
    print(json.dumps(summary_rows[0] if summary_rows else {}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
