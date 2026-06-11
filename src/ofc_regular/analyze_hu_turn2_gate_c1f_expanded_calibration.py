"""Gate C1f expanded calibration for HU Turn2 Stage8 candidates.

This is calibration-only. It evaluates fixed threshold candidates on the C1e
teacher-EV cache and chooses candidates for a later seat-swap gate. It does not
start C2-small, 50k teacher generation, T1 training, or production training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .analyze_hu_turn2_gate_c1_large_calibration import sample_source_group
from .analyze_hu_turn2_gate_c1_followup import enriched_state_rows, load_action_original_indices
from .analyze_hu_turn2_gate_c1d_threshold_repair import (
    c1d_strategy_specs,
    threshold_passes_strategy,
)
from .analyze_hu_turn2_pilot_calibration import load_model, rows_for_split, write_csv
from .train_hu_turn2_pilot_model import load_cache, predict_all, target_matrix
from .train_torch_action_value import select_device


DEFAULT_CACHE_DIR = Path("D:/ofc-pineapple-storage/regular-ofc-pineapple/feature_cache/hu_t2_stage8_20k_mc512")
DEFAULT_MODEL = Path("models/hu_turn2_stage8_broad_20k_mc512_reference_override_cached_rank_wide.pt")
DEFAULT_C1E_DIR = Path("outputs/hu_turn2_stage1_pilot_training_c1e_teacher_ev_feature_cache")
DEFAULT_OUTPUT_DIR = Path("outputs/hu_turn2_stage1_pilot_training_c1f_expanded_calibration")
PRIMARY_CANDIDATES = {
    "confidence_lcb196_m2p5_r0_g0p7",
    "confidence_lcb164_m2p5_r0_g0p7",
    "source_filtered_lcb196",
    "best_diagnostic_m2p5_r0_g0p7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--c1e-dir", type=Path, default=DEFAULT_C1E_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32768)
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fixed_candidate_specs() -> list[dict[str, Any]]:
    specs = [spec for spec in c1d_strategy_specs() if spec["candidate_id"] in PRIMARY_CANDIDATES]
    specs.extend(
        [
            {
                "candidate_id": "confidence_lcb196_m2p25_r0_g0p7",
                "threshold_family": "lcb95_conservative_relaxed_margin",
                "min_margin": 2.25,
                "reference_min_margin": 0.0,
                "gate_threshold": 0.7,
                "requires_current_lcb_1p96_positive": True,
            },
            {
                "candidate_id": "confidence_lcb164_m2p25_r0_g0p7",
                "threshold_family": "lcb90_medium_relaxed_margin",
                "min_margin": 2.25,
                "reference_min_margin": 0.0,
                "gate_threshold": 0.7,
                "requires_current_lcb_1p64_positive": True,
            },
            {
                "candidate_id": "confidence_lcb196_m2p0_r0_g0p75",
                "threshold_family": "lcb95_conservative_relaxed_margin_tighter_gate",
                "min_margin": 2.0,
                "reference_min_margin": 0.0,
                "gate_threshold": 0.75,
                "requires_current_lcb_1p96_positive": True,
            },
            {
                "candidate_id": "confidence_lcb164_m2p0_r0_g0p75",
                "threshold_family": "lcb90_medium_relaxed_margin_tighter_gate",
                "min_margin": 2.0,
                "reference_min_margin": 0.0,
                "gate_threshold": 0.75,
                "requires_current_lcb_1p64_positive": True,
            },
        ]
    )
    return specs


def load_c1e_index(c1e_dir: Path) -> dict[int, dict[str, str]]:
    rows = read_csv(c1e_dir / "teacher_ev_feature_cache_merged.csv")
    return {safe_int(row.get("state_index")): row for row in rows}


def attach_c1e_fields(rows: list[dict[str, Any]], c1e_rows: dict[int, dict[str, str]]) -> None:
    for row in rows:
        c1e = c1e_rows.get(safe_int(row.get("state_index")), {})
        row["c1e_split"] = c1e.get("c1e_split") or (
            "unbiased" if sample_source_group(row) == "policy_on_distribution" else "enriched"
        )
        row["position"] = c1e.get("position") or row.get("seat", "unknown")
        row["replay_ready"] = safe_int(c1e.get("replay_ready", 0))
        row["source"] = c1e.get("source") or row.get("source_bucket", "")
        row["source_bucket"] = c1e.get("source_bucket") or row.get("source_bucket", "")
        row["teacher_gain_mc512"] = row["actual_delta_candidate_vs_baseline"]
        row["false_positive_flag"] = int(safe_float(row["actual_delta_candidate_vs_baseline"]) < 0.0)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def teacher_positive(row: dict[str, Any]) -> bool:
    delta = safe_float(row.get("actual_delta_best_vs_baseline"))
    se = safe_float(row.get("SE_delta"))
    return delta >= 0.25 and (se <= 0.0 or delta >= 2.0 * se)


def metric_summary(candidate_id: str, family: str, evaluated: list[dict[str, Any]], fired: list[dict[str, Any]]) -> dict[str, Any]:
    gains = [safe_float(row["actual_delta_candidate_vs_baseline"]) for row in fired]
    losses = [max(0.0, -gain) for gain in gains]
    predicted_delta = [safe_float(row["predicted_delta_vs_baseline"]) for row in fired]
    gate = [safe_float(row["gate_probability"]) for row in fired]
    lcb96 = [safe_float(row["gain_lcb_1p96"]) for row in fired]
    lcb64 = [safe_float(row["gain_lcb_1p64"]) for row in fired]
    positives = [row for row in evaluated if teacher_positive(row)]
    missed_positive = [row for row in positives if row not in fired]
    source_counts = Counter(sample_source_group(row) for row in fired)
    run_bucket_counts = Counter(str(row.get("run_bucket", "unknown")) for row in fired)
    position_counts = Counter(str(row.get("position") or row.get("seat", "unknown")) for row in fired)
    return {
        "candidate_id": candidate_id,
        "threshold_family": family,
        "rows": len(evaluated),
        "fires": len(fired),
        "fire_rate": len(fired) / max(len(evaluated), 1),
        "avg_gain": float(np.mean(gains)) if gains else 0.0,
        "median_gain": float(np.median(gains)) if gains else 0.0,
        "false_positive_count": sum(1 for gain in gains if gain < 0.0),
        "false_positive_rate": sum(1 for gain in gains if gain < 0.0) / max(len(gains), 1),
        "avg_false_positive_cost": float(np.mean([loss for loss in losses if loss > 0.0])) if any(loss > 0.0 for loss in losses) else 0.0,
        "p90_loss": percentile(losses, 90),
        "p95_loss": percentile(losses, 95),
        "p99_loss": percentile(losses, 99),
        "max_loss": max(losses) if losses else 0.0,
        "positive_fire_count": sum(1 for gain in gains if gain > 0.0),
        "negative_fire_count": sum(1 for gain in gains if gain < 0.0),
        "positive_opportunity_count": len(positives),
        "missed_positive_count": len(missed_positive),
        "missed_positive_rate": len(missed_positive) / max(len(positives), 1),
        "predicted_delta_mean": float(np.mean(predicted_delta)) if predicted_delta else 0.0,
        "predicted_delta_p50": percentile(predicted_delta, 50),
        "predicted_delta_p90": percentile(predicted_delta, 90),
        "gate_probability_mean": float(np.mean(gate)) if gate else 0.0,
        "gate_probability_p50": percentile(gate, 50),
        "lcb_1p96_mean": float(np.mean(lcb96)) if lcb96 else 0.0,
        "lcb_1p64_mean": float(np.mean(lcb64)) if lcb64 else 0.0,
        "unbiased_fires": sum(1 for row in fired if row.get("c1e_split") == "unbiased"),
        "enriched_fires": sum(1 for row in fired if row.get("c1e_split") == "enriched"),
        "first_fires": position_counts.get("first", 0),
        "second_fires": position_counts.get("second", 0),
        "source_counts": json.dumps(dict(source_counts), sort_keys=True),
        "run_bucket_counts": json.dumps(dict(run_bucket_counts), sort_keys=True),
        "position_counts": json.dumps(dict(position_counts), sort_keys=True),
        "dominant_source_share": max(source_counts.values(), default=0) / max(len(fired), 1),
        "dominant_position_share": max(position_counts.values(), default=0) / max(len(fired), 1),
    }


def fired_for_strategy(rows: list[dict[str, Any]], strategy: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in rows if threshold_passes_strategy(row, strategy)]


def candidate_metrics(rows: list[dict[str, Any]], strategies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for strategy in strategies:
        fired = fired_for_strategy(rows, strategy)
        output.append(metric_summary(strategy["candidate_id"], strategy["threshold_family"], rows, fired))
    return output


def group_specs(rows: list[dict[str, Any]]) -> list[tuple[str, str, list[dict[str, Any]]]]:
    specs: list[tuple[str, str, list[dict[str, Any]]]] = []
    for value in ("unbiased", "enriched"):
        specs.append(("c1e_split", value, [row for row in rows if row.get("c1e_split") == value]))
    for bucket in sorted({str(row.get("run_bucket", "unknown")) for row in rows}):
        specs.append(("run_bucket", bucket, [row for row in rows if str(row.get("run_bucket", "unknown")) == bucket]))
    for field in ("actual_high_regret", "actual_low_margin", "actual_teacher_disagreement"):
        specs.append(("actual_flag", field, [row for row in rows if bool(row.get(field))]))
        specs.append(("actual_flag", f"not_{field}", [row for row in rows if not bool(row.get(field))]))
    return specs


def source_breakdown(rows: list[dict[str, Any]], strategies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for strategy in strategies:
        for group_type, group, subset in group_specs(rows):
            fired = fired_for_strategy(subset, strategy)
            summary = metric_summary(strategy["candidate_id"], strategy["threshold_family"], subset, fired)
            output.append(
                {
                    "candidate_id": strategy["candidate_id"],
                    "threshold_family": strategy["threshold_family"],
                    "group_type": group_type,
                    "group": group,
                    "rows": len(subset),
                    "fires": summary["fires"],
                    "fire_rate": summary["fire_rate"],
                    "avg_gain": summary["avg_gain"],
                    "false_positive_rate": summary["false_positive_rate"],
                    "p95_loss": summary["p95_loss"],
                    "max_loss": summary["max_loss"],
                }
            )
    return output


def position_breakdown(rows: list[dict[str, Any]], strategies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for strategy in strategies:
        for position in ("first", "second"):
            subset = [row for row in rows if str(row.get("position") or row.get("seat")) == position]
            fired = fired_for_strategy(subset, strategy)
            summary = metric_summary(strategy["candidate_id"], strategy["threshold_family"], subset, fired)
            output.append(
                {
                    "candidate_id": strategy["candidate_id"],
                    "threshold_family": strategy["threshold_family"],
                    "position": position,
                    "rows": len(subset),
                    "fires": summary["fires"],
                    "fire_rate": summary["fire_rate"],
                    "avg_gain": summary["avg_gain"],
                    "false_positive_rate": summary["false_positive_rate"],
                    "p95_loss": summary["p95_loss"],
                    "max_loss": summary["max_loss"],
                }
            )
    return output


def decile_rows(rows: list[dict[str, Any]], strategies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    variables = (
        ("predicted_delta", "predicted_delta_vs_baseline"),
        ("lcb_1p96", "gain_lcb_1p96"),
        ("lcb_1p64", "gain_lcb_1p64"),
        ("gate_probability", "gate_probability"),
    )
    for strategy in strategies:
        fired_ids = {safe_int(row["state_index"]) for row in fired_for_strategy(rows, strategy)}
        for variable, field in variables:
            values = np.asarray([safe_float(row.get(field)) for row in rows], dtype=np.float64)
            if values.size == 0:
                continue
            order = np.argsort(values, kind="mergesort")
            ranks = np.empty_like(order)
            ranks[order] = np.arange(values.size)
            deciles = np.minimum((ranks * 10) // max(values.size, 1), 9)
            for decile in range(10):
                subset = [row for i, row in enumerate(rows) if int(deciles[i]) == decile]
                fired = [row for row in subset if safe_int(row["state_index"]) in fired_ids]
                gains = [safe_float(row["actual_delta_candidate_vs_baseline"]) for row in fired]
                all_candidate_gains = [safe_float(row["actual_delta_candidate_vs_baseline"]) for row in subset]
                output.append(
                    {
                        "candidate_id": strategy["candidate_id"],
                        "threshold_family": strategy["threshold_family"],
                        "variable": variable,
                        "decile": decile,
                        "rows": len(subset),
                        "fires": len(fired),
                        "fire_rate": len(fired) / max(len(subset), 1),
                        "value_min": min((safe_float(row.get(field)) for row in subset), default=0.0),
                        "value_max": max((safe_float(row.get(field)) for row in subset), default=0.0),
                        "teacher_gain_mean_all_rows": float(np.mean(all_candidate_gains)) if all_candidate_gains else 0.0,
                        "teacher_gain_mean_fired": float(np.mean(gains)) if gains else 0.0,
                        "false_positive_rate_fired": sum(1 for gain in gains if gain < 0.0) / max(len(gains), 1),
                        "teacher_positive_rate_all_rows": sum(1 for row in subset if teacher_positive(row)) / max(len(subset), 1),
                    }
                )
    return output


def negative_control_rows(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    control = next((row for row in metrics if row["candidate_id"] == "best_diagnostic_m2p5_r0_g0p7"), None)
    if not control:
        return []
    output: list[dict[str, Any]] = []
    for row in metrics:
        if row["candidate_id"] == control["candidate_id"]:
            continue
        output.append(
            {
                "candidate_id": row["candidate_id"],
                "control_candidate_id": control["candidate_id"],
                "candidate_fires": row["fires"],
                "control_fires": control["fires"],
                "candidate_fp_rate": row["false_positive_rate"],
                "control_fp_rate": control["false_positive_rate"],
                "fp_rate_reduction": safe_float(control["false_positive_rate"]) - safe_float(row["false_positive_rate"]),
                "candidate_avg_gain": row["avg_gain"],
                "control_avg_gain": control["avg_gain"],
                "avg_gain_delta": safe_float(row["avg_gain"]) - safe_float(control["avg_gain"]),
                "candidate_p95_loss": row["p95_loss"],
                "control_p95_loss": control["p95_loss"],
                "candidate_unbiased_fires": row["unbiased_fires"],
                "candidate_enriched_fires": row["enriched_fires"],
            }
        )
    return output


def candidate_status(row: dict[str, Any]) -> tuple[str, str]:
    blockers: list[str] = []
    fires = safe_int(row["fires"])
    if fires < 15:
        blockers.append("fires_lt_15")
    if safe_float(row["false_positive_rate"]) > 0.10:
        blockers.append("false_positive_rate_gt_10pct")
    if safe_float(row["avg_gain"]) <= 0.0:
        blockers.append("avg_gain_not_positive")
    if safe_float(row["p95_loss"]) > 1.25:
        blockers.append("p95_loss_gt_1p25")
    if safe_float(row["p99_loss"]) > 1.25:
        blockers.append("p99_loss_gt_1p25")
    if safe_int(row["first_fires"]) == 0 or safe_int(row["second_fires"]) == 0:
        blockers.append("position_one_sided")
    if safe_int(row["unbiased_fires"]) == 0:
        blockers.append("unbiased_fires_zero")
    if safe_float(row["dominant_source_share"]) >= 0.80 and fires > 0:
        blockers.append("source_biased")

    core_ok = (
        fires >= 15
        and safe_float(row["false_positive_rate"]) <= 0.10
        and safe_float(row["avg_gain"]) > 0.0
        and safe_float(row["p95_loss"]) <= 1.25
    )
    if core_ok and not blockers:
        return "go", ""
    if core_ok:
        return "conditional_source_biased", ";".join(blockers)
    return "no_go", ";".join(blockers)


def recommended_candidates(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in metrics:
        if row["candidate_id"] == "best_diagnostic_m2p5_r0_g0p7":
            continue
        status, blockers = candidate_status(row)
        output.append(
            {
                "candidate_id": row["candidate_id"],
                "seat_swap_status": status,
                "blockers": blockers,
                "fires": row["fires"],
                "unbiased_fires": row["unbiased_fires"],
                "enriched_fires": row["enriched_fires"],
                "first_fires": row["first_fires"],
                "second_fires": row["second_fires"],
                "avg_gain": row["avg_gain"],
                "false_positive_rate": row["false_positive_rate"],
                "p95_loss": row["p95_loss"],
                "p99_loss": row["p99_loss"],
                "max_loss": row["max_loss"],
            }
        )
    primary_order = {
        "confidence_lcb196_m2p5_r0_g0p7": 0,
        "confidence_lcb164_m2p5_r0_g0p7": 1,
        "source_filtered_lcb196": 2,
    }
    primary = [row for row in output if row["candidate_id"] in primary_order]
    optional = [row for row in output if row["candidate_id"] not in primary_order]
    primary.sort(
        key=lambda row: (
            {"go": 0, "conditional_source_biased": 1, "no_go": 2}.get(str(row["seat_swap_status"]), 3),
            primary_order.get(str(row["candidate_id"]), 99),
        )
    )
    optional.sort(
        key=lambda row: (
            {"go": 0, "conditional_source_biased": 1, "no_go": 2}.get(str(row["seat_swap_status"]), 3),
            safe_float(row["false_positive_rate"]),
            -safe_int(row["fires"]),
            -safe_float(row["avg_gain"]),
        )
    )
    selected = [row for row in primary if row["seat_swap_status"] != "no_go"]
    selected.extend(row for row in optional if row["seat_swap_status"] != "no_go")
    if len(selected) < 4:
        selected.extend(row for row in primary + optional if row["seat_swap_status"] == "no_go")
    return selected[:4]


def c2_decision(recommended: list[dict[str, Any]]) -> tuple[str, list[str]]:
    blockers: list[str] = []
    viable = [row for row in recommended if row["seat_swap_status"] in {"go", "conditional_source_biased"}]
    if not viable:
        blockers.append("no_viable_candidate")
    if all(safe_int(row["unbiased_fires"]) == 0 for row in viable):
        blockers.append("unbiased_fires_zero_for_all_viable_candidates")
    if all("source_biased" in str(row.get("blockers", "")) for row in viable):
        blockers.append("all_viable_candidates_source_biased")
    if len(viable) < 1:
        blockers.append("seat_swap_candidate_count_lt_1")
    return ("Go" if not blockers else "No-Go", blockers)


def write_summary(
    output_dir: Path,
    *,
    metrics: list[dict[str, Any]],
    recommended: list[dict[str, Any]],
    c2_status: str,
    c2_blockers: list[str],
    c1e_rows: int,
    replay_ready_rows: int,
    elapsed: float,
) -> None:
    lines = [
        "# HU Turn2 Stage8 Gate C1f Expanded Calibration",
        "",
        "C1f is calibration-only. It does not authorize C2-small, 50k teacher, T1, production training, or production runtime changes.",
        "",
        f"- C1e rows loaded: `{c1e_rows}`",
        f"- C1e replay-ready rows: `{replay_ready_rows}`",
        f"- elapsed seconds: `{elapsed:.2f}`",
        f"- C1f expanded calibration: `completed`",
        f"- C2-small seat-swap: `{c2_status}`",
        f"- C2-small blockers: `{';'.join(c2_blockers) if c2_blockers else 'none'}`",
        "- runtime caveat: `LCB filters in this analyzer use MC512 teacher gain LCB; they are calibration filters, not directly deployable runtime gates.`",
        "",
        "## Candidate Metrics",
        "",
        "| candidate | fires | unbiased | enriched | first | second | FP rate | avg gain | p95 loss | p99 loss | max loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            "| {candidate_id} | {fires} | {unbiased_fires} | {enriched_fires} | {first_fires} | {second_fires} | {false_positive_rate:.3f} | {avg_gain:.4f} | {p95_loss:.4f} | {p99_loss:.4f} | {max_loss:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Recommended Seat-Swap Candidates",
            "",
            "| candidate | status | blockers | fires | avg gain | FP rate |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in recommended:
        lines.append(
            "| {candidate_id} | {seat_swap_status} | {blockers} | {fires} | {avg_gain:.4f} | {false_positive_rate:.3f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Decisions",
            "",
            "- production training: `No-Go`",
            "- T1 training: `No-Go`",
            "- 50k teacher: `No-Go`",
            f"- C2-small heldout: `{c2_status}`",
        ]
    )
    (output_dir / "c1f_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "go_nogo_for_c2_small.md").write_text(
        "\n".join(
            [
                "# Go / No-Go For C2-Small",
                "",
                f"- C2-small: `{c2_status}`",
                f"- blockers: `{';'.join(c2_blockers) if c2_blockers else 'none'}`",
                "- 50k teacher: `No-Go`",
                "- T1 training: `No-Go`",
                "- production training: `No-Go`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    started_at = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    c1e_index = load_c1e_index(args.c1e_dir)
    device = select_device(torch, args.device)
    cache = load_cache(args.cache_dir)
    net, stats, _payload = load_model(torch, args.model, device)
    predictions = predict_all(torch, net, cache, stats, device, args.batch_size)
    targets = target_matrix(cache)
    action_original_indices = load_action_original_indices(args.cache_dir, int(cache["metadata"]["action_count"]))
    rows = enriched_state_rows(cache, predictions, targets, action_original_indices)
    attach_c1e_fields(rows, c1e_index)

    strategies = fixed_candidate_specs()
    metrics = candidate_metrics(rows, strategies)
    source_rows = source_breakdown(rows, strategies)
    position_rows = position_breakdown(rows, strategies)
    calibration_rows = decile_rows(rows, strategies)
    negative_rows = negative_control_rows(metrics)
    recommended = recommended_candidates(metrics)
    c2_status, c2_blockers = c2_decision(recommended)
    replay_ready_rows = sum(safe_int(row.get("replay_ready")) for row in rows)

    write_csv(args.output_dir / "c1f_candidate_metrics.csv", metrics)
    write_csv(args.output_dir / "c1f_source_breakdown.csv", source_rows)
    write_csv(args.output_dir / "c1f_position_breakdown.csv", position_rows)
    write_csv(args.output_dir / "c1f_calibration_deciles.csv", calibration_rows)
    write_csv(args.output_dir / "c1f_negative_control_comparison.csv", negative_rows)
    write_csv(args.output_dir / "c1f_recommended_seat_swap_candidates.csv", recommended)
    write_summary(
        args.output_dir,
        metrics=metrics,
        recommended=recommended,
        c2_status=c2_status,
        c2_blockers=c2_blockers,
        c1e_rows=len(c1e_index),
        replay_ready_rows=replay_ready_rows,
        elapsed=time.perf_counter() - started_at,
    )
    print(
        json.dumps(
            {
                "gate": "C1f",
                "rows": len(rows),
                "c1e_rows": len(c1e_index),
                "replay_ready_rows": replay_ready_rows,
                "c2_small": c2_status,
                "c2_blockers": c2_blockers,
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
