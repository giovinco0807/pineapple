"""Aggregate GCP shards for HU Turn2 Stage8 C3 larger seat-swap validation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .evaluate_hu_turn2_stage8_seat_swap import aggregate_seed_rows, write_csv, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Downloaded GCP run directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), q / 100.0))


def collect_result_dirs(input_dir: Path) -> list[Path]:
    roots = [input_dir / "results", input_dir]
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("hu_turn2_stage8_20k_seed_breakdown.csv"):
            found.append(path.parent)
    return sorted(set(found))


def aggregate_grid(seed_rows: list[dict[str, Any]], shard_grid_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregated = aggregate_seed_rows(seed_rows)
    teacher_by_config: dict[str, dict[str, Any]] = {}
    for row in shard_grid_rows:
        config_id = str(row.get("config_id", ""))
        if config_id and config_id not in teacher_by_config:
            teacher_by_config[config_id] = {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "config_id",
                    "hu_turn2_min_margin",
                    "hu_turn2_reference_min_margin",
                    "hu_turn2_gate_threshold",
                    "paired_seeds",
                    "hands",
                    "aggregate_ev_per_hand",
                    "std_error_seed_means",
                    "ci95_low_seed_means",
                    "ci95_high_seed_means",
                    "seed_count",
                    "decision_count",
                    "override_count",
                    "runtime_override_rate",
                    "paired_seed_wins",
                    "paired_seed_losses",
                    "paired_seed_ties",
                }
            }
    output: list[dict[str, Any]] = []
    for row in aggregated:
        combined = dict(row)
        combined.update(teacher_by_config.get(str(row.get("config_id", "")), {}))
        output.append(combined)
    output.sort(key=lambda item: safe_float(item.get("aggregate_ev_per_hand")), reverse=True)
    return output


def aggregate_position(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("config_id", "")), str(row.get("seat", "")))].append(row)
    output: list[dict[str, Any]] = []
    for (config_id, seat), group in sorted(grouped.items()):
        hands = sum(int(safe_float(row.get("hands"))) for row in group)
        weighted_ev = sum(safe_float(row.get("ev_per_hand")) * int(safe_float(row.get("hands"))) for row in group)
        overrides = sum(int(safe_float(row.get("override_count"))) for row in group)
        decisions = sum(int(safe_float(row.get("decision_count"))) for row in group)
        output.append(
            {
                "config_id": config_id,
                "seat": seat,
                "hands": hands,
                "ev_per_hand": weighted_ev / max(hands, 1),
                "override_count": overrides,
                "decision_count": decisions,
                "override_rate": overrides / max(decisions, 1),
            }
        )
    return output


def runtime_distribution(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for config_id in sorted({str(row.get("config_id", "")) for row in decisions}):
        group = [row for row in decisions if str(row.get("config_id", "")) == config_id]
        fired = [row for row in group if bool(row.get("override_fired"))]
        no_override = Counter(str(row.get("no_override_reason", "")) for row in group if not bool(row.get("override_fired")))
        for field in ("predicted_delta", "gate_probability", "reference_margin_raw", "model_score", "runtime_latency_ms"):
            values = [safe_float(row.get(field)) for row in fired if row.get(field) not in (None, "")]
            output.append(
                {
                    "config_id": config_id,
                    "metric": field,
                    "fired_count": len(fired),
                    "decision_count": len(group),
                    "override_rate": len(fired) / max(len(group), 1),
                    "mean": float(np.mean(values)) if values else 0.0,
                    "median": float(np.median(values)) if values else 0.0,
                    "p10": percentile(values, 10),
                    "p90": percentile(values, 90),
                    "p95": percentile(values, 95),
                    "min": min(values) if values else 0.0,
                    "max": max(values) if values else 0.0,
                    "no_override_reason_counts": json.dumps(dict(no_override), sort_keys=True),
                }
            )
    return output


def failure_rows(decisions: list[dict[str, Any]], limit: int = 30) -> list[dict[str, Any]]:
    failures = [row for row in decisions if bool(row.get("override_fired"))]
    failures.sort(key=lambda row: safe_float(row.get("candidate_seat_score")), reverse=False)
    output: list[dict[str, Any]] = []
    for row in failures[:limit]:
        output.append(
            {
                "config_id": row.get("config_id"),
                "seed": row.get("seed"),
                "hand_id": row.get("hand_id"),
                "hand_seed": row.get("hand_seed"),
                "seat": row.get("seat"),
                "seat_swap": row.get("seat_swap"),
                "candidate_seat_score": safe_float(row.get("candidate_seat_score")),
                "hero_board": row.get("hero_board"),
                "opponent_board": row.get("opponent_board"),
                "dead_cards": row.get("dead_cards"),
                "cards_to_place": row.get("cards_to_place"),
                "baseline_action": row.get("baseline_action"),
                "stage8_action": row.get("stage8_action"),
                "final_action": row.get("final_action"),
                "predicted_delta": safe_float(row.get("predicted_delta")),
                "gate_probability": safe_float(row.get("gate_probability")),
                "reference_margin_raw": safe_float(row.get("reference_margin_raw")),
                "model_score": safe_float(row.get("model_score")),
                "failure_label": classify_runtime_failure(row),
            }
        )
    return output


def classify_runtime_failure(row: dict[str, Any]) -> str:
    score = safe_float(row.get("candidate_seat_score"))
    if safe_float(row.get("predicted_delta")) < 3.0:
        return "false_positive_gate"
    if safe_float(row.get("gate_probability")) < 0.925:
        return "low_gate_confidence"
    if score < -20.0:
        return "tail_loss_audit_required"
    return "other"


def write_summary(path: Path, grid_rows: list[dict[str, Any]], manifest: dict[str, Any], *, elapsed_status: dict[str, Any]) -> None:
    lines = [
        "# HU Turn2 Stage8 C3 Larger Seat-Swap Validation",
        "",
        "C3 is validation-only. It does not authorize 50k teacher, T1, production training, or production runtime changes.",
        "",
        "- T3 continuation: `Stage7_candidate_A_m5_r10`",
        "- Stage8 mode: `HU T2 selective override`, not full replacement",
        "- seed_stride: `{}`".format(manifest.get("seed_stride", "")),
        "- games_per_seed: `{}`".format(manifest.get("games_per_seed", "")),
        "- completed_shards: `{}`".format(elapsed_status.get("completed_shards", "")),
        "",
        "## Seat-Swap Results",
        "",
        "| config | EV/hand | CI low | CI high | seeds | paired | overrides | override rate | teacher avg gain | FP | p95 loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in grid_rows:
        lines.append(
            "| {config_id} | {ev:.4f} | {low:.4f} | {high:.4f} | {seeds} | {paired} | {overrides} | {rate:.4f} | {gain:.4f} | {fp:.4f} | {p95:.4f} |".format(
                config_id=row.get("config_id", ""),
                ev=safe_float(row.get("aggregate_ev_per_hand")),
                low=safe_float(row.get("ci95_low_seed_means")),
                high=safe_float(row.get("ci95_high_seed_means")),
                seeds=row.get("seed_count", ""),
                paired=row.get("paired_seeds", ""),
                overrides=row.get("override_count", ""),
                rate=safe_float(row.get("runtime_override_rate")),
                gain=safe_float(row.get("avg_gain_on_override")),
                fp=safe_float(row.get("false_positive_override_rate")),
                p95=safe_float(row.get("p95_loss")),
            )
        )
    best = grid_rows[0] if grid_rows else {}
    ci_low = safe_float(best.get("ci95_low_seed_means"))
    best_ev = safe_float(best.get("aggregate_ev_per_hand"))
    c3_go = best_ev > 0.0 and ci_low > -0.05
    next_step = (
        "selected MC4096/8192 refinement before any production decision"
        if c3_go
        else "revise runtime proxy gate; use high-MC only as diagnostic audit for fired/top-loss states"
    )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- C3 larger seat-swap: `{'Go' if c3_go else 'No-Go'}`",
            "- production: `No-Go`",
            "- 50k teacher: `No-Go`",
            "- T1 training: `No-Go`",
            f"- next step: `{next_step}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    result_dirs = collect_result_dirs(args.input_dir)
    if not result_dirs:
        raise SystemExit(f"no shard result dirs found under {args.input_dir}")

    seed_rows: list[dict[str, Any]] = []
    grid_shard_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for result_dir in result_dirs:
        seed_rows.extend(read_csv(result_dir / "hu_turn2_stage8_20k_seed_breakdown.csv"))
        grid_shard_rows.extend(read_csv(result_dir / "hu_turn2_stage8_20k_threshold_grid_results.csv"))
        position_rows.extend(read_csv(result_dir / "hu_turn2_stage8_20k_position_breakdown.csv"))
        bucket_rows.extend(read_csv(result_dir / "hu_turn2_stage8_20k_bucket_breakdown.csv"))
        decisions.extend(iter_jsonl(result_dir / "hu_turn2_stage8_20k_runtime_decisions.jsonl"))

    grid_rows = aggregate_grid(seed_rows, grid_shard_rows)
    position = aggregate_position(position_rows)
    runtime_dist = runtime_distribution(decisions)
    failures = failure_rows(decisions)
    status = {
        "completed_shards": len(result_dirs),
        "decision_rows": len(decisions),
        "override_rows": sum(1 for row in decisions if bool(row.get("override_fired"))),
    }

    write_csv(args.output_dir / "c3_seat_swap_results.csv", grid_rows)
    write_csv(args.output_dir / "c3_seed_breakdown.csv", seed_rows)
    write_csv(args.output_dir / "c3_position_breakdown.csv", position)
    write_csv(args.output_dir / "c3_source_breakdown.csv", bucket_rows)
    write_csv(args.output_dir / "c3_runtime_distribution.csv", runtime_dist)
    write_jsonl(args.output_dir / "c3_failure_top30.jsonl", failures)
    write_summary(args.output_dir / "c3_summary.md", grid_rows, manifest, elapsed_status=status)
    c3_go = bool(grid_rows and safe_float(grid_rows[0].get("aggregate_ev_per_hand")) > 0.0 and safe_float(grid_rows[0].get("ci95_low_seed_means")) > -0.05)
    (args.output_dir / "c3_go_nogo.md").write_text(
        "\n".join(
            [
                "# HU T2 Stage8 C3 Go / No-Go",
                "",
                f"- C3: `{'Go' if c3_go else 'No-Go'}`",
                "- production: `No-Go`",
                "- 50k teacher: `No-Go`",
                "- T1 training: `No-Go`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if c3_go:
        recommendation = [
            "C3 is positive enough to justify selected MC4096/8192 refinement on proxy-fired, near-threshold, and top-loss states.",
            "Do not start production, T1, or 50k teacher until refinement and another larger validation pass are clean.",
        ]
    else:
        recommendation = [
            "C3 is not positive. Do not start 50k teacher, T1, production training, or production runtime work.",
            "Next work should revise the runtime proxy gate and use high-MC only as a diagnostic audit for fired/top-loss states.",
        ]
    (args.output_dir / "c3_recommended_next_step.md").write_text(
        "\n".join(["# Recommended Next Step", "", *recommendation]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "completed_shards": len(result_dirs), "c3_go": c3_go}, indent=2))


if __name__ == "__main__":
    main()
