"""Analyze HU T1 TopK confirm decision logs.

The confirm MC delta is a gate diagnostic. Strength claims must use realized
seat-swap counterfactual deltas on valid fired decisions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

PERFORMANCE_METRIC_SOURCE = "realized_seat_swap_counterfactual"
CONFIRM_DELTA_METRIC_ROLE = "gate_diagnostic_only"
CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED = False


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stderr(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def read_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
                rows.append(row)
    return rows


def group_by_config(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("config_id") or "unknown")].append(row)
    return dict(grouped)


def summarize_config(config_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [
        row
        for row in rows
        if row.get("realized_delta_valid")
        and row.get("realized_candidate_seat_delta") not in (None, "")
    ]
    fired = [row for row in valid if bool(row.get("override_fired"))]
    fired_invalid = [row for row in rows if bool(row.get("override_fired")) and not row.get("realized_delta_valid")]
    non_fired = [row for row in valid if not bool(row.get("override_fired"))]
    fired_deltas = [safe_float(row.get("realized_candidate_seat_delta")) for row in fired]
    confirm_deltas = [safe_float(row.get("confirm_delta")) for row in fired]
    confirm_ses = [safe_float(row.get("confirm_delta_se")) for row in fired]
    losses = [max(0.0, -delta) for delta in fired_deltas]
    runtime = [safe_float(row.get("runtime_latency_ms")) for row in rows]
    confirm_runtime = [safe_float(row.get("confirm_mc_latency_ms")) for row in rows]
    rerank_runtime = [safe_float(row.get("mc_rerank_latency_ms")) for row in rows]
    non_fired_deltas = [safe_float(row.get("realized_candidate_seat_delta")) for row in non_fired]
    non_fired_nonzero = [delta for delta in non_fired_deltas if abs(delta) > 1e-9]
    final_match_counts = Counter(final_action_match_status(row) for row in non_fired)
    fired_mean = mean(fired_deltas)
    fired_se = stderr(fired_deltas)
    override_rate_valid = len(fired) / max(len(valid), 1)
    return {
        "config_id": config_id,
        "decision_count": len(rows),
        "valid_decision_count": len(valid),
        "override_count": sum(1 for row in rows if bool(row.get("override_fired"))),
        "valid_override_count": len(fired),
        "invalid_override_count": len(fired_invalid),
        "override_rate_valid": override_rate_valid,
        "realized_per_fire_delta_mean": fired_mean,
        "realized_per_fire_delta_se": fired_se,
        "realized_per_fire_ci95_low": fired_mean - 1.96 * fired_se,
        "realized_per_fire_ci95_high": fired_mean + 1.96 * fired_se,
        "estimated_ev_per_decision": override_rate_valid * fired_mean,
        "estimated_ev_per_decision_ci95_low": override_rate_valid * (fired_mean - 1.96 * fired_se),
        "estimated_ev_per_decision_ci95_high": override_rate_valid * (fired_mean + 1.96 * fired_se),
        "loss_count": sum(1 for delta in fired_deltas if delta < -1e-9),
        "win_count": sum(1 for delta in fired_deltas if delta > 1e-9),
        "zero_count": sum(1 for delta in fired_deltas if abs(delta) <= 1e-9),
        "p90_loss": percentile(losses, 90),
        "p95_loss": percentile(losses, 95),
        "p99_loss": percentile(losses, 99),
        "max_loss": max(losses, default=0.0),
        "confirm_delta_mean_on_valid_fired": mean(confirm_deltas),
        "confirm_delta_se_mean_on_valid_fired": mean(confirm_ses),
        "confirm_delta_metric_role": CONFIRM_DELTA_METRIC_ROLE,
        "confirm_delta_performance_claim_allowed": CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED,
        "performance_metric_source": PERFORMANCE_METRIC_SOURCE,
        "non_fired_valid_count": len(non_fired),
        "non_fired_nonzero_count": len(non_fired_nonzero),
        "non_fired_counterfactual_nonzero_count": len(non_fired_nonzero),
        "non_fired_final_matches_baseline_count": final_match_counts["match"],
        "non_fired_final_mismatch_count": final_match_counts["mismatch"],
        "non_fired_final_match_unknown_count": final_match_counts["unknown"],
        "non_fired_delta_sum": sum(non_fired_deltas),
        "non_fired_delta_max_abs": max((abs(delta) for delta in non_fired_deltas), default=0.0),
        "runtime_latency_ms_mean": mean(runtime),
        "runtime_latency_ms_p95": percentile(runtime, 95),
        "mc_rerank_latency_ms_mean": mean(rerank_runtime),
        "confirm_mc_latency_ms_mean": mean(confirm_runtime),
        "no_override_reason_counts": json.dumps(dict(Counter(str(row.get("no_override_reason") or "") for row in rows)), sort_keys=True),
    }


def final_action_match_status(row: dict[str, Any]) -> str:
    final_index = row.get("final_action_index")
    baseline_index = row.get("baseline_action_index")
    if final_index is not None and baseline_index is not None:
        return "match" if final_index == baseline_index else "mismatch"
    final_action = row.get("final_action")
    baseline_action = row.get("baseline_action")
    if final_action is not None and baseline_action is not None:
        return "match" if final_action == baseline_action else "mismatch"
    return "unknown"


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = [
        summarize_config(config_id, group)
        for config_id, group in group_by_config(rows).items()
    ]
    summaries.sort(key=lambda row: safe_float(row.get("estimated_ev_per_decision")), reverse=True)
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], *, input_paths: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# HU T1 TopK Confirm Analysis",
        "",
        f"- inputs: `{', '.join(str(path) for path in input_paths)}`",
        f"- performance metric source: `{PERFORMANCE_METRIC_SOURCE}`",
        f"- confirm delta role: `{CONFIRM_DELTA_METRIC_ROLE}`",
        f"- confirm delta performance claim allowed: `{CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED}`",
        "",
        "| config | decisions | valid fired | fired mean | fired CI95 | est EV/decision | non-fired cf nonzero | non-fired final mismatch | p95 loss | reason counts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {config} | {decisions} | {fired} | {mean:.4f} | [{low:.4f},{high:.4f}] | {ev:.4f} | {nf_cf} | {nf_mismatch} | {p95:.4f} | `{reasons}` |".format(
                config=row["config_id"],
                decisions=int(row["decision_count"]),
                fired=int(row["valid_override_count"]),
                mean=safe_float(row["realized_per_fire_delta_mean"]),
                low=safe_float(row["realized_per_fire_ci95_low"]),
                high=safe_float(row["realized_per_fire_ci95_high"]),
                ev=safe_float(row["estimated_ev_per_decision"]),
                nf_cf=int(row["non_fired_counterfactual_nonzero_count"]),
                nf_mismatch=int(row["non_fired_final_mismatch_count"]),
                p95=safe_float(row["p95_loss"]),
                reasons=row["no_override_reason_counts"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    summary_rows = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "hu_turn1_topk_confirm_summary.csv", summary_rows)
    write_markdown(args.output_dir / "hu_turn1_topk_confirm_summary.md", summary_rows, input_paths=args.input)
    manifest = {
        "schema": "hu_turn1_topk_confirm_analysis_v1",
        "input": [str(path) for path in args.input],
        "rows": len(rows),
        "configs": len(summary_rows),
        "performance_metric_source": PERFORMANCE_METRIC_SOURCE,
        "confirm_delta_metric_role": CONFIRM_DELTA_METRIC_ROLE,
        "confirm_delta_performance_claim_allowed": CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
