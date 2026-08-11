"""Extract Stage8c fire+veto replay targets from scored TopK rows.

This tool selects rows for additional independent replay. It does not evaluate
runtime performance. A row is eligible only when its realized/local replay delta
is still unknown, so follow-up replay can provide fresh evidence instead of
reusing gate or confirm estimates.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .extract_hu_turn2_stage8c_topk_replay_targets import (
    replay_missing_fields,
    row_key,
    safe_float,
    safe_int,
    truthy,
)


DEFAULT_OUTPUT_DIR = Path("outputs/training/hu_turn2_stage8c_fire_veto_replay_targets")
DEFAULT_TARGET_NAME = "fire_veto_replay_targets.jsonl"
DEFAULT_FIRE_THRESHOLDS = (0.8, 0.85, 0.9)
DEFAULT_VETO_THRESHOLDS = (0.4, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distillation-jsonl", type=Path, required=True)
    parser.add_argument("--fire-predictions-csv", type=Path, required=True)
    parser.add_argument("--veto-predictions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fire-thresholds", default=",".join(str(value) for value in DEFAULT_FIRE_THRESHOLDS))
    parser.add_argument("--veto-thresholds", default=",".join(str(value) for value in DEFAULT_VETO_THRESHOLDS))
    parser.add_argument("--fire-boundary-width", type=float, default=0.05)
    parser.add_argument("--veto-boundary-width", type=float, default=0.05)
    parser.add_argument(
        "--exclude-candidate-source",
        action="append",
        default=[],
        help="Candidate source to exclude from replay target extraction. Repeat to exclude multiple sources.",
    )
    parser.add_argument(
        "--max-veto-probability",
        type=float,
        default=None,
        help="Skip rows whose veto probability is at or above this value. Missing veto predictions are not filtered.",
    )
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="Fire prediction split to include. Repeat for multiple splits. Defaults to all.",
    )
    parser.add_argument("--max-targets", type=int, default=0)
    return parser.parse_args()


def parse_thresholds(raw: str) -> list[float]:
    values = sorted({float(part.strip()) for part in str(raw or "").split(",") if part.strip()})
    if not values:
        raise ValueError("at least one threshold is required")
    return values


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def has_observed_delta(row: dict[str, Any]) -> bool:
    if truthy(row.get("realized_delta_observed")):
        return True
    if str(row.get("local_replay_status", "")).lower() == "ok":
        return True
    if str(row.get("local_replay_label", "")).strip():
        return True
    return False


def index_by_row_key(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str, str, str, str, str, str, str], dict[str, Any]]:
    return {row_key(row): row for row in rows}


def selected_pairs(
    *,
    fire_probability: float,
    veto_probability: float | None,
    fire_thresholds: list[float],
    veto_thresholds: list[float],
    fire_boundary_width: float,
    veto_boundary_width: float,
) -> tuple[list[dict[str, Any]], set[str]]:
    pairs: list[dict[str, Any]] = []
    reasons: set[str] = set()
    for fire_threshold in fire_thresholds:
        fire_selected = fire_probability >= fire_threshold
        fire_boundary = abs(fire_probability - fire_threshold) <= fire_boundary_width
        for veto_threshold in veto_thresholds:
            pair_reasons: set[str] = set()
            if veto_probability is None:
                if fire_selected:
                    pair_reasons.add("fire_selected_missing_veto_prediction")
                if fire_boundary:
                    pair_reasons.add("fire_boundary_missing_veto_prediction")
            else:
                veto_kept = veto_probability < veto_threshold
                veto_boundary = abs(veto_probability - veto_threshold) <= veto_boundary_width
                if fire_selected and veto_kept:
                    pair_reasons.add("kept_after_fire_veto")
                    if veto_probability >= veto_threshold - veto_boundary_width:
                        pair_reasons.add("just_kept_by_veto")
                if fire_selected and not veto_kept and veto_probability <= veto_threshold + veto_boundary_width:
                    pair_reasons.add("just_vetoed_by_veto")
                if fire_boundary:
                    pair_reasons.add("fire_boundary")
                if fire_selected and veto_boundary:
                    pair_reasons.add("veto_boundary")
            if pair_reasons:
                pairs.append(
                    {
                        "fire_threshold": fire_threshold,
                        "veto_threshold": veto_threshold,
                        "reasons": ",".join(sorted(pair_reasons)),
                    }
                )
                reasons.update(pair_reasons)
    return pairs, reasons


def target_priority(row: dict[str, Any]) -> tuple[float, float, float]:
    reasons = set(str(row.get("replay_target_reasons", "")).split(","))
    reason_rank = 0
    if "kept_after_fire_veto" in reasons:
        reason_rank = 3
    elif "just_kept_by_veto" in reasons or "just_vetoed_by_veto" in reasons:
        reason_rank = 2
    elif "fire_selected_missing_veto_prediction" in reasons:
        reason_rank = 1
    return (
        float(reason_rank),
        safe_float(row.get("fire_probability")),
        -safe_float(row.get("veto_probability"), 99.0),
    )


def build_target(
    *,
    source_row: dict[str, Any],
    fire_prediction: dict[str, str],
    veto_prediction: dict[str, str] | None,
    pairs: list[dict[str, Any]],
    reasons: set[str],
) -> dict[str, Any]:
    fire_probability = safe_float(fire_prediction.get("risk_probability"))
    veto_probability = None if veto_prediction is None else safe_float(veto_prediction.get("risk_probability"))
    missing = replay_missing_fields(source_row)
    target = dict(source_row)
    target.update(
        {
            "schema": "hu_turn2_stage8c_fire_veto_replay_target_v1",
            "replay_target_reason": "fire_veto_boundary_or_kept_unknown_delta",
            "replay_target_reasons": ",".join(sorted(reasons)),
            "replay_target_threshold_pairs": pairs,
            "replay_target_threshold_pairs_csv": ";".join(
                f"f{pair['fire_threshold']:g}_v{pair['veto_threshold']:g}:{pair['reasons']}" for pair in pairs
            ),
            "fire_probability": fire_probability,
            "fire_prediction_split": fire_prediction.get("split", ""),
            "fire_prediction_label": safe_int(fire_prediction.get("label"), -1),
            "veto_probability": "" if veto_probability is None else veto_probability,
            "veto_prediction_missing": veto_prediction is None,
            "veto_prediction_split": "" if veto_prediction is None else veto_prediction.get("split", ""),
            "veto_prediction_label": "" if veto_prediction is None else safe_int(veto_prediction.get("label"), -1),
            "realized_delta_observed": False,
            "observed_performance_claim": "No",
            "replay_ready": not missing,
            "replay_blocker": ",".join(missing),
            "production_p2_fixed": "No-Go",
            "teacher_50k": "No-Go",
            "t1_training": "No-Go",
            "note": (
                "Selected for independent replay by fire+veto probabilities. "
                "Do not use as performance evidence before replay."
            ),
        }
    )
    return target


def extract_fire_veto_targets(
    distillation_rows: list[dict[str, Any]],
    fire_rows: list[dict[str, str]],
    veto_rows: list[dict[str, str]],
    *,
    fire_thresholds: list[float],
    veto_thresholds: list[float],
    fire_boundary_width: float,
    veto_boundary_width: float,
    splits: set[str] | None = None,
    excluded_candidate_sources: set[str] | None = None,
    max_veto_probability: float | None = None,
    max_targets: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    distillation_index = index_by_row_key(distillation_rows)
    veto_index = index_by_row_key(veto_rows)
    target_by_key: dict[tuple[str, str, str, str, str, str, str, str], dict[str, Any]] = {}
    threshold_counters: Counter[tuple[float, float, str]] = Counter()
    summary = Counter()

    for fire_prediction in fire_rows:
        split = str(fire_prediction.get("split", ""))
        if splits is not None and split not in splits:
            summary["filtered_split"] += 1
            continue
        key = row_key(fire_prediction)
        source_row = distillation_index.get(key)
        if source_row is None:
            summary["missing_distillation_row"] += 1
            continue
        if has_observed_delta(source_row):
            summary["observed_delta_skipped"] += 1
            continue
        if excluded_candidate_sources is not None and str(source_row.get("candidate_source", "")) in excluded_candidate_sources:
            summary["filtered_candidate_source"] += 1
            continue
        veto_prediction = veto_index.get(key)
        if (
            max_veto_probability is not None
            and veto_prediction is not None
            and safe_float(veto_prediction.get("risk_probability")) >= max_veto_probability
        ):
            summary["filtered_veto_probability"] += 1
            continue
        pairs, reasons = selected_pairs(
            fire_probability=safe_float(fire_prediction.get("risk_probability")),
            veto_probability=None if veto_prediction is None else safe_float(veto_prediction.get("risk_probability")),
            fire_thresholds=fire_thresholds,
            veto_thresholds=veto_thresholds,
            fire_boundary_width=fire_boundary_width,
            veto_boundary_width=veto_boundary_width,
        )
        if not pairs:
            summary["not_selected"] += 1
            continue
        target = build_target(
            source_row=source_row,
            fire_prediction=fire_prediction,
            veto_prediction=veto_prediction,
            pairs=pairs,
            reasons=reasons,
        )
        previous = target_by_key.get(key)
        if previous is None or target_priority(target) > target_priority(previous):
            target_by_key[key] = target
        summary["selected_prediction_rows"] += 1
        summary["selected_with_veto_prediction" if veto_prediction is not None else "selected_missing_veto_prediction"] += 1
        for pair in pairs:
            for reason in pair["reasons"].split(","):
                threshold_counters[(float(pair["fire_threshold"]), float(pair["veto_threshold"]), reason)] += 1

    targets = sorted(
        target_by_key.values(),
        key=lambda row: (
            -target_priority(row)[0],
            -safe_float(row.get("fire_probability")),
            safe_float(row.get("veto_probability"), 99.0),
            str(row.get("hand_seed", "")),
            str(row.get("action_signature", "")),
        ),
    )
    if max_targets > 0:
        targets = targets[:max_targets]
    threshold_rows = [
        {"fire_threshold": fire, "veto_threshold": veto, "reason": reason, "rows": count}
        for (fire, veto, reason), count in sorted(threshold_counters.items())
    ]
    breakdown = build_breakdown_rows(targets)
    summary_dict = dict(summary)
    summary_dict.update(
        {
            "distillation_rows": len(distillation_rows),
            "fire_prediction_rows": len(fire_rows),
            "veto_prediction_rows": len(veto_rows),
            "targets": len(targets),
            "replay_ready_targets": sum(1 for row in targets if truthy(row.get("replay_ready"))),
            "targets_with_veto_prediction": sum(1 for row in targets if not truthy(row.get("veto_prediction_missing"))),
            "targets_missing_veto_prediction": sum(1 for row in targets if truthy(row.get("veto_prediction_missing"))),
        }
    )
    return targets, threshold_rows, breakdown, summary_dict


def build_breakdown_rows(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, str, list[dict[str, Any]]]] = [("overall", "all", targets)]
    for field in (
        "seat",
        "recommended_training_use",
        "candidate_source",
        "fire_prediction_split",
        "veto_prediction_split",
        "veto_prediction_missing",
    ):
        for value in sorted({str(row.get(field, "")) for row in targets}):
            groups.append((field, value, [row for row in targets if str(row.get(field, "")) == value]))
    reason_values = sorted(
        {
            reason
            for row in targets
            for reason in str(row.get("replay_target_reasons", "")).split(",")
            if reason
        }
    )
    for reason in reason_values:
        groups.append(
            (
                "replay_target_reason_member",
                reason,
                [row for row in targets if reason in str(row.get("replay_target_reasons", "")).split(",")],
            )
        )
    for field, value, subset in groups:
        if not subset:
            continue
        fire_probs = [safe_float(row.get("fire_probability")) for row in subset]
        veto_probs = [
            safe_float(row.get("veto_probability"))
            for row in subset
            if row.get("veto_probability") not in (None, "")
        ]
        rows.append(
            {
                "group_field": field,
                "group_value": value,
                "rows": len(subset),
                "replay_ready_rows": sum(1 for row in subset if truthy(row.get("replay_ready"))),
                "with_veto_prediction": sum(1 for row in subset if not truthy(row.get("veto_prediction_missing"))),
                "fire_probability_mean": sum(fire_probs) / len(fire_probs),
                "fire_probability_min": min(fire_probs),
                "fire_probability_max": max(fire_probs),
                "veto_probability_mean": sum(veto_probs) / len(veto_probs) if veto_probs else "",
                "veto_probability_min": min(veto_probs) if veto_probs else "",
                "veto_probability_max": max(veto_probs) if veto_probs else "",
            }
        )
    return rows


def write_summary(path: Path, summary: dict[str, Any], threshold_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# HU T2 Stage8c Fire + Veto Replay Targets",
        "",
        "This is replay prep only. It does not claim runtime performance.",
        "",
        f"- targets: `{summary.get('targets', 0)}`",
        f"- replay-ready targets: `{summary.get('replay_ready_targets', 0)}`",
        f"- targets with veto prediction: `{summary.get('targets_with_veto_prediction', 0)}`",
        f"- targets missing veto prediction: `{summary.get('targets_missing_veto_prediction', 0)}`",
        f"- observed-delta rows skipped: `{summary.get('observed_delta_skipped', 0)}`",
        "- Production / P2 fixed / 50k / T1: `No-Go`",
        "",
        "## Threshold Reason Counts",
        "",
        "| fire | veto | reason | rows |",
        "|---:|---:|---|---:|",
    ]
    for row in threshold_rows:
        lines.append(
            f"| {float(row['fire_threshold']):g} | {float(row['veto_threshold']):g} | "
            f"{row['reason']} | {int(row['rows'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Replay targets must be evaluated with independent paired rollouts before being used as evidence.",
            "- `veto_prediction_missing=true` means the veto head was not scored for that row; generate full-row veto predictions before relying on combo filtering.",
            "- Confirm/MC gate averages remain diagnostics only; realized or independent replay deltas are the evaluation signal.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.max_targets < 0:
        raise ValueError("--max-targets must be non-negative")
    fire_thresholds = parse_thresholds(args.fire_thresholds)
    veto_thresholds = parse_thresholds(args.veto_thresholds)
    if args.fire_boundary_width < 0.0 or args.veto_boundary_width < 0.0:
        raise ValueError("boundary widths must be non-negative")
    splits = {str(value) for value in args.split} if args.split else None
    excluded_candidate_sources = {
        str(value).strip() for value in args.exclude_candidate_source if str(value).strip()
    } or None
    targets, threshold_rows, breakdown, summary = extract_fire_veto_targets(
        read_jsonl(args.distillation_jsonl),
        read_csv(args.fire_predictions_csv),
        read_csv(args.veto_predictions_csv),
        fire_thresholds=fire_thresholds,
        veto_thresholds=veto_thresholds,
        fire_boundary_width=args.fire_boundary_width,
        veto_boundary_width=args.veto_boundary_width,
        splits=splits,
        excluded_candidate_sources=excluded_candidate_sources,
        max_veto_probability=args.max_veto_probability,
        max_targets=args.max_targets,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / DEFAULT_TARGET_NAME, targets)
    write_csv(args.output_dir / "fire_veto_replay_target_thresholds.csv", threshold_rows)
    write_csv(args.output_dir / "fire_veto_replay_target_breakdown.csv", breakdown)
    (args.output_dir / "fire_veto_replay_target_manifest.json").write_text(
        json.dumps(
            {
                "schema": "hu_turn2_stage8c_fire_veto_replay_target_manifest_v1",
                "distillation_jsonl": str(args.distillation_jsonl),
                "fire_predictions_csv": str(args.fire_predictions_csv),
                "veto_predictions_csv": str(args.veto_predictions_csv),
                "output_dir": str(args.output_dir),
                "fire_thresholds": fire_thresholds,
                "veto_thresholds": veto_thresholds,
                "fire_boundary_width": args.fire_boundary_width,
                "veto_boundary_width": args.veto_boundary_width,
                "splits": "all" if splits is None else sorted(splits),
                "excluded_candidate_sources": [] if excluded_candidate_sources is None else sorted(excluded_candidate_sources),
                "max_veto_probability": args.max_veto_probability,
                **summary,
                "production_p2_fixed": "No-Go",
                "teacher_50k": "No-Go",
                "t1_training": "No-Go",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_summary(args.output_dir / "fire_veto_replay_target_summary.md", summary, threshold_rows)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
