"""Gate C1 large-calibration audit for HU Turn2 threshold candidates.

This gate is deliberately evaluation-only. It can be run on the current 2k
pilot cache as a readiness replay, but it only authorizes a 50k teacher pass
when the supplied cache contains the requested 5k-10k calibration sample.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .analyze_hu_turn2_pilot_calibration import (
    load_model,
    percentile_summary,
    rows_for_split,
    state_rows,
    threshold_metrics,
    write_csv,
)
from .train_hu_turn2_pilot_model import load_cache, predict_all, target_matrix
from .train_torch_action_value import select_device

C1_MIN_MARGIN_GRID = (1.5, 2.0, 2.5, 3.0)
C1_REFERENCE_MARGIN_GRID = (0.0, 0.5, 1.0)
C1_GATE_THRESHOLD_GRID = (0.6, 0.7, 0.8, 0.9)
C1_SPLITS = ("val", "test", "holdout")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("D:/ofc-pineapple-storage/regular-ofc-pineapple/feature_cache/hu_t2_stage1_pilot_2000_mc512"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/hu_turn2_stage8_pilot_2000_mc512_reference_override_cached_rank_wide.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/hu_turn2_gate_c1_large_calibration"),
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--threshold-split", choices=("val", "test", "holdout"), default="test")
    parser.add_argument("--target-min-states", type=int, default=5000)
    parser.add_argument("--target-max-states", type=int, default=10000)
    return parser.parse_args()


def sample_source_group(row: dict[str, Any]) -> str:
    raw = str(row.get("run_bucket") or row.get("bucket_group") or "unknown")
    if raw == "natural":
        return "policy_on_distribution"
    if raw == "random_off_policy":
        return "random_off_policy"
    if "low_margin" in raw:
        return "near_threshold_spots"
    if "high_regret" in raw or "teacher_disagreement" in raw:
        return "difficult_spots"
    return "other"


def baseline_confidence_group(row: dict[str, Any]) -> str:
    reference_margin = float(row.get("reference_margin_raw", 0.0) or 0.0)
    if reference_margin >= 1.0:
        return "high"
    if reference_margin >= 0.5:
        return "medium"
    return "low"


def fired_rows(
    rows: list[dict[str, Any]],
    *,
    split_name: str,
    min_margin: float,
    reference_min_margin: float,
    gate_threshold: float,
) -> list[dict[str, Any]]:
    subset = rows_for_split(rows, split_name)
    return [
        row
        for row in subset
        if not row["candidate_is_baseline"]
        and float(row["predicted_delta_vs_baseline"]) >= min_margin
        and float(row["reference_margin_raw"]) >= reference_min_margin
        and float(row["gate_probability"]) >= gate_threshold
    ]


def extended_threshold_metrics(
    rows: list[dict[str, Any]],
    *,
    split_name: str,
    min_margin: float,
    reference_min_margin: float,
    gate_threshold: float,
    sample_status: str,
    target_min_states: int,
    target_max_states: int,
) -> dict[str, Any]:
    base = threshold_metrics(
        rows,
        split_name=split_name,
        min_delta=min_margin,
        min_reference=reference_min_margin,
        gate_threshold=gate_threshold,
        threshold_kind="gate_c1_absolute",
    )
    fired = fired_rows(
        rows,
        split_name=split_name,
        min_margin=min_margin,
        reference_min_margin=reference_min_margin,
        gate_threshold=gate_threshold,
    )
    source_counts = Counter(sample_source_group(row) for row in fired)
    seat_counts = Counter(str(row.get("seat", "unknown")) for row in fired)
    confidence_counts = Counter(baseline_confidence_group(row) for row in fired)
    override_count = len(fired)
    dominant_source_share = max(source_counts.values(), default=0) / max(override_count, 1)
    dominant_seat_share = max(seat_counts.values(), default=0) / max(override_count, 1)
    base.update(
        {
            "gate": "C1",
            "sample_status": sample_status,
            "target_min_states": target_min_states,
            "target_max_states": target_max_states,
            "source_counts": json.dumps(dict(source_counts), sort_keys=True),
            "seat_counts": json.dumps(dict(seat_counts), sort_keys=True),
            "baseline_confidence_counts": json.dumps(dict(confidence_counts), sort_keys=True),
            "dominant_source_share": dominant_source_share,
            "dominant_seat_share": dominant_seat_share,
            "source_biased": int(override_count > 0 and dominant_source_share >= 0.80),
            "position_biased": int(override_count > 0 and dominant_seat_share >= 0.80),
            "balanced_first_second": int(seat_counts.get("first", 0) > 0 and seat_counts.get("second", 0) > 0),
        }
    )
    return base


def threshold_grid_rows(
    rows: list[dict[str, Any]],
    *,
    sample_status: str,
    target_min_states: int,
    target_max_states: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split_name in C1_SPLITS:
        for min_margin in C1_MIN_MARGIN_GRID:
            for reference_min_margin in C1_REFERENCE_MARGIN_GRID:
                for gate_threshold in C1_GATE_THRESHOLD_GRID:
                    output.append(
                        extended_threshold_metrics(
                            rows,
                            split_name=split_name,
                            min_margin=min_margin,
                            reference_min_margin=reference_min_margin,
                            gate_threshold=gate_threshold,
                            sample_status=sample_status,
                            target_min_states=target_min_states,
                            target_max_states=target_max_states,
                        )
                    )
    return output


def source_position_breakdown_rows(rows: list[dict[str, Any]], grid_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for config in grid_rows:
        split_name = str(config["split"])
        min_margin = float(config["hu_turn2_min_margin"])
        reference_min_margin = float(config["hu_turn2_reference_min_margin"])
        gate_threshold = float(config["gate_threshold"])
        subset = rows_for_split(rows, split_name)
        fired = fired_rows(
            rows,
            split_name=split_name,
            min_margin=min_margin,
            reference_min_margin=reference_min_margin,
            gate_threshold=gate_threshold,
        )
        groups = sorted(
            {
                (sample_source_group(row), str(row.get("seat", "unknown")), baseline_confidence_group(row))
                for row in subset
            }
        )
        for source_group, seat, confidence in groups:
            group_total = [
                row
                for row in subset
                if sample_source_group(row) == source_group
                and str(row.get("seat", "unknown")) == seat
                and baseline_confidence_group(row) == confidence
            ]
            group_fired = [
                row
                for row in fired
                if sample_source_group(row) == source_group
                and str(row.get("seat", "unknown")) == seat
                and baseline_confidence_group(row) == confidence
            ]
            gains = [float(row["actual_delta_candidate_vs_baseline"]) for row in group_fired]
            losses = [max(0.0, -gain) for gain in gains]
            output.append(
                {
                    "split": split_name,
                    "hu_turn2_min_margin": min_margin,
                    "hu_turn2_reference_min_margin": reference_min_margin,
                    "gate_threshold": gate_threshold,
                    "source_group": source_group,
                    "seat": seat,
                    "baseline_confidence": confidence,
                    "evaluated_states": len(group_total),
                    "override_count": len(group_fired),
                    "override_rate": len(group_fired) / max(len(group_total), 1),
                    "teacher_avg_gain_on_override": float(np.mean(gains)) if gains else 0.0,
                    "false_positive_count": sum(1 for gain in gains if gain < 0.0),
                    "false_positive_rate": sum(1 for gain in gains if gain < 0.0) / max(len(gains), 1),
                    "worst_override_loss": max(losses) if losses else 0.0,
                }
            )
    return output


def worst_override_rows(rows: list[dict[str, Any]], grid_rows: list[dict[str, Any]], *, limit: int = 200) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for config in grid_rows:
        split_name = str(config["split"])
        min_margin = float(config["hu_turn2_min_margin"])
        reference_min_margin = float(config["hu_turn2_reference_min_margin"])
        gate_threshold = float(config["gate_threshold"])
        for row in fired_rows(
            rows,
            split_name=split_name,
            min_margin=min_margin,
            reference_min_margin=reference_min_margin,
            gate_threshold=gate_threshold,
        ):
            actual_gain = float(row["actual_delta_candidate_vs_baseline"])
            output.append(
                {
                    "split": split_name,
                    "state_index": row["state_index"],
                    "hu_turn2_min_margin": min_margin,
                    "hu_turn2_reference_min_margin": reference_min_margin,
                    "gate_threshold": gate_threshold,
                    "source_group": sample_source_group(row),
                    "run_bucket": row.get("run_bucket", ""),
                    "bucket_group": row.get("bucket_group", ""),
                    "seat": row.get("seat", ""),
                    "baseline_confidence": baseline_confidence_group(row),
                    "predicted_delta_vs_baseline": row["predicted_delta_vs_baseline"],
                    "reference_margin_raw": row["reference_margin_raw"],
                    "gate_probability": row["gate_probability"],
                    "actual_delta_candidate_vs_baseline": actual_gain,
                    "candidate_false_positive": int(actual_gain < 0.0),
                    "candidate_loss": max(0.0, -actual_gain),
                    "teacher_best_margin": row["teacher_best_margin"],
                    "action_count": row["action_count"],
                }
            )
    output.sort(
        key=lambda row: (
            float(row["actual_delta_candidate_vs_baseline"]),
            -float(row["candidate_loss"]),
            float(row["hu_turn2_min_margin"]),
            float(row["hu_turn2_reference_min_margin"]),
            float(row["gate_threshold"]),
        )
    )
    return output[:limit]


def threshold_shortlist_rows(grid_rows: list[dict[str, Any]], *, threshold_split: str, sample_status: str) -> list[dict[str, Any]]:
    split_rows = [row for row in grid_rows if row["split"] == threshold_split]
    candidates = [
        row
        for row in split_rows
        if 0.01 <= float(row["override_rate"]) <= 0.05
        and float(row["teacher_avg_gain_on_override"]) > 0.0
        and int(row["false_positive_count"]) == 0
        and float(row["max_loss"]) <= 1e-12
    ]
    candidates.sort(
        key=lambda row: (
            int(row["source_biased"]),
            int(row["position_biased"]),
            -int(row["balanced_first_second"]),
            -float(row["teacher_avg_gain_on_override"]),
            abs(float(row["override_rate"]) - 0.03),
            -float(row["gate_threshold"]),
        )
    )
    if not candidates:
        candidates = [
            row
            for row in split_rows
            if float(row["override_rate"]) > 0.0 and float(row["teacher_avg_gain_on_override"]) > 0.0
        ]
        candidates.sort(
            key=lambda row: (
                float(row["false_positive_rate"]),
                -float(row["teacher_avg_gain_on_override"]),
                abs(float(row["override_rate"]) - 0.03),
            )
        )
    output: list[dict[str, Any]] = []
    for rank, row in enumerate(candidates[:12], start=1):
        source_biased = bool(int(row["source_biased"]))
        position_biased = bool(int(row["position_biased"]))
        if sample_status != "target_met":
            status = "diagnostic_only_sample_below_c1_target"
        elif source_biased or position_biased:
            status = "diagnostic_only_distribution_biased"
        else:
            status = "c2_shortlist_candidate"
        output.append(
            {
                "rank": rank,
                "recommended_for_c2": int(status == "c2_shortlist_candidate" and rank <= 3),
                "shortlist_status": status,
                **row,
            }
        )
    return output


def best_metric_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    viable = [row for row in rows if float(row["override_rate"]) > 0.0 and float(row["teacher_avg_gain_on_override"]) > 0.0]
    if not viable:
        return None
    viable.sort(
        key=lambda row: (
            int(row["source_biased"]),
            int(row["position_biased"]),
            float(row["false_positive_rate"]),
            -float(row["teacher_avg_gain_on_override"]),
            abs(float(row["override_rate"]) - 0.03),
        )
    )
    return viable[0]


def write_calibration_summary(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    grid_rows: list[dict[str, Any]],
    shortlist_rows: list[dict[str, Any]],
    threshold_split: str,
    sample_status: str,
    target_min_states: int,
    target_max_states: int,
) -> None:
    subset = rows_for_split(rows, threshold_split)
    pred_stats = percentile_summary(float(row["predicted_delta_vs_baseline"]) for row in subset)
    ref_stats = percentile_summary(float(row["reference_margin_raw"]) for row in subset)
    margin_stats = percentile_summary(float(row["teacher_best_margin"]) for row in subset)
    split_grid = [row for row in grid_rows if row["split"] == threshold_split]
    best = best_metric_row(split_grid)
    lines = [
        "# HU Turn2 Gate C1 Large Calibration",
        "",
        "This gate is threshold calibration only. It does not authorize production training.",
        "",
        "## Sample Status",
        "",
        f"- target calibration sample: `{target_min_states}`-`{target_max_states}` states",
        f"- actual loaded states: `{len(rows)}`",
        f"- threshold split: `{threshold_split}` with `{len(subset)}` states",
        f"- C1 sample status: `{sample_status}`",
        "",
    ]
    if sample_status != "target_met":
        lines.extend(
            [
                "The current run is a pilot replay/readiness check because the loaded cache is below the 5k C1 target.",
                "Do not treat this as a completed Gate C1 decision.",
                "",
            ]
        )
    lines.extend(
        [
            "## Scale",
            "",
            f"- predicted_delta_vs_baseline p50/p90/p95/max: `{pred_stats['p50']:.4f}` / `{pred_stats['p90']:.4f}` / `{pred_stats['p95']:.4f}` / `{pred_stats['max']:.4f}`",
            f"- reference_margin_raw p50/p90/p95/max: `{ref_stats['p50']:.4f}` / `{ref_stats['p90']:.4f}` / `{ref_stats['p95']:.4f}` / `{ref_stats['max']:.4f}`",
            f"- teacher_best_margin p50/p90/p95/max: `{margin_stats['p50']:.4f}` / `{margin_stats['p90']:.4f}` / `{margin_stats['p95']:.4f}` / `{margin_stats['max']:.4f}`",
            "",
            "## Grid",
            "",
            f"- min_margin grid: `{list(C1_MIN_MARGIN_GRID)}`",
            f"- reference_min_margin grid: `{list(C1_REFERENCE_MARGIN_GRID)}`",
            f"- gate_threshold grid: `{list(C1_GATE_THRESHOLD_GRID)}`",
            f"- threshold rows written: `{len(grid_rows)}`",
            f"- shortlist rows written: `{len(shortlist_rows)}`",
            "",
        ]
    )
    if best:
        lines.extend(
            [
                "## Best Diagnostic Row",
                "",
                f"- threshold: `m{float(best['hu_turn2_min_margin']):.2f}_r{float(best['hu_turn2_reference_min_margin']):.2f}_g{float(best['gate_threshold']):.2f}`",
                f"- override: `{int(best['override_count'])}/{int(best['evaluated_states'])}` = `{float(best['override_rate']):.4f}`",
                f"- teacher avg gain: `{float(best['teacher_avg_gain_on_override']):.4f}`",
                f"- false positives: `{int(best['false_positive_count'])}`, rate `{float(best['false_positive_rate']):.4f}`",
                f"- source biased: `{bool(int(best['source_biased']))}`, position biased: `{bool(int(best['position_biased']))}`",
                f"- source counts: `{best['source_counts']}`",
                f"- seat counts: `{best['seat_counts']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "",
            "- production training: `No-Go`",
            (
                "- 50k teacher pass: `Conditional-Go candidate pending C2`"
                if sample_status == "target_met" and any(int(row["recommended_for_c2"]) for row in shortlist_rows)
                else "- 50k teacher pass: `No-Go/Pending until true 5k-10k C1 sample is evaluated`"
            ),
            "- T1/main production training remains blocked.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_go_nogo(
    path: Path,
    *,
    sample_status: str,
    shortlist_rows: list[dict[str, Any]],
) -> None:
    c2_ready = sample_status == "target_met" and any(int(row["recommended_for_c2"]) for row in shortlist_rows)
    lines = [
        "# Gate C1 Go/No-Go For 50k",
        "",
        "- production candidate training: `No-Go`",
        f"- C1 sample status: `{sample_status}`",
        f"- C2-ready shortlist rows: `{sum(1 for row in shortlist_rows if int(row['recommended_for_c2']))}`",
        "",
        "## Decision",
        "",
        (
            "50k teacher pass is `Conditional-Go` after freezing the C2 shortlist and rerunning on a separate seed/split."
            if c2_ready
            else "50k teacher pass is `No-Go/Pending`; complete the 5k-10k Gate C1 sample before launching 50k."
        ),
        "",
        "## Guardrails",
        "",
        "- Do not start T1 or production candidate training from this gate.",
        "- Do not use `min_margin=5/reference_min_margin=10`; that scale is too conservative for the observed T2 pilot distribution.",
        "- Keep first/second and source attribution in every downstream threshold report.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started_at = time.time()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    device = select_device(torch, args.device)
    cache = load_cache(args.cache_dir.resolve())
    targets = target_matrix(cache)
    net, stats, model_payload = load_model(torch, args.model.resolve(), device)
    predictions = predict_all(torch, net, cache, stats, device, args.batch_size)
    rows = state_rows(cache, predictions, targets)

    sample_status = "target_met" if len(rows) >= args.target_min_states else "pilot_replay_only_insufficient_sample"
    grid_rows = threshold_grid_rows(
        rows,
        sample_status=sample_status,
        target_min_states=args.target_min_states,
        target_max_states=args.target_max_states,
    )
    breakdown = source_position_breakdown_rows(rows, grid_rows)
    worst_rows = worst_override_rows(rows, grid_rows)
    shortlist = threshold_shortlist_rows(grid_rows, threshold_split=args.threshold_split, sample_status=sample_status)

    write_csv(output_dir / "threshold_grid_metrics.csv", grid_rows)
    write_csv(output_dir / "threshold_shortlist.csv", shortlist)
    write_csv(output_dir / "source_position_breakdown.csv", breakdown)
    write_csv(output_dir / "worst_overrides.csv", worst_rows)
    write_calibration_summary(
        output_dir / "calibration_summary.md",
        rows=rows,
        grid_rows=grid_rows,
        shortlist_rows=shortlist,
        threshold_split=args.threshold_split,
        sample_status=sample_status,
        target_min_states=args.target_min_states,
        target_max_states=args.target_max_states,
    )
    write_go_nogo(output_dir / "go_nogo_for_50k.md", sample_status=sample_status, shortlist_rows=shortlist)

    manifest = {
        "schema": "hu_turn2_gate_c1_large_calibration_v1",
        "cache_dir": str(args.cache_dir.resolve()),
        "model": str(args.model.resolve()),
        "output_dir": str(output_dir),
        "device": device,
        "threshold_split": args.threshold_split,
        "states": len(rows),
        "actions": int(cache["metadata"]["action_count"]),
        "target_min_states": args.target_min_states,
        "target_max_states": args.target_max_states,
        "sample_status": sample_status,
        "threshold_grid": {
            "min_margin": list(C1_MIN_MARGIN_GRID),
            "reference_min_margin": list(C1_REFERENCE_MARGIN_GRID),
            "gate_threshold": list(C1_GATE_THRESHOLD_GRID),
        },
        "wrote": [
            "calibration_summary.md",
            "threshold_grid_metrics.csv",
            "threshold_shortlist.csv",
            "source_position_breakdown.csv",
            "worst_overrides.csv",
            "go_nogo_for_50k.md",
            "gate_c1_manifest.json",
        ],
        "production_candidate_training_allowed": False,
        "teacher_50k_launched": False,
    }
    (output_dir / "gate_c1_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "schema": "hu_turn2_gate_c1_large_calibration_run_v1",
        "model_kind": model_payload.get("model_kind"),
        "states": len(rows),
        "actions": int(cache["metadata"]["action_count"]),
        "sample_status": sample_status,
        "shortlist_rows": len(shortlist),
        "elapsed_seconds": time.time() - started_at,
    }
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
