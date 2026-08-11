"""Extract replay targets selected by a Stage8c TopK distillation risk head.

This is an evaluation-prep tool. It does not treat non-fired rows as measured
performance. Rows selected by the pre-confirm gate but without observed
realized deltas are emitted as replay targets so they can be independently
evaluated before any runtime-gate claim is made.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT_DIR = Path("outputs/training/hu_turn2_stage8c_topk_confirm_replay_targets")
DEFAULT_TARGET_NAME = "topk_confirm_unknown_replay_targets.jsonl"
REPLAY_REQUIRED_FIELDS = (
    "hero_board",
    "opponent_board",
    "dead_cards",
    "cards_to_place",
    "baseline_action",
    "candidate_action",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distillation-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--threshold",
        type=float,
        action="append",
        default=None,
        help="Risk probability threshold. Can be repeated. Defaults to 0.8.",
    )
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="Prediction split to include. Repeat for multiple splits. Defaults to all splits.",
    )
    parser.add_argument(
        "--exclude-recommended-use",
        action="append",
        default=[],
        help="Skip source rows whose recommended_training_use matches this value. Can be repeated.",
    )
    parser.add_argument("--max-targets", type=int, default=0, help="Optional cap after sorting by risk_probability.")
    return parser.parse_args()


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


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalize_path(value: Any) -> str:
    return str(value or "").replace("/", "\\").lower()


def row_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str, str, str]:
    return (
        normalize_path(row.get("source_log")),
        str(row.get("config_id", "")),
        str(row.get("hand_seed", "")),
        str(row.get("seat", "")),
        str(row.get("state_signature", "")),
        str(row.get("action_signature", "")),
        str(safe_int(row.get("candidate_index", row.get("candidate_action_index")), -1)),
        str(safe_int(row.get("baseline_index", row.get("baseline_action_index")), -1)),
    )


def replay_missing_fields(row: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for field in REPLAY_REQUIRED_FIELDS:
        if row.get(field) in (None, "", [], {}):
            missing.append(field)
    return missing


def load_distillation_index(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str, str, str, str, str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str, str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        index[row_key(row)] = row
    return index


def filter_distillation_rows(
    rows: Iterable[dict[str, Any]],
    *,
    excluded_recommended_uses: Iterable[str],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    excluded = {str(value).strip() for value in excluded_recommended_uses if str(value).strip()}
    kept: list[dict[str, Any]] = []
    removed: Counter[str] = Counter()
    for row in rows:
        use = str(row.get("recommended_training_use") or row.get("target_recommended_use") or "")
        if use in excluded:
            removed[use] += 1
            continue
        kept.append(row)
    return kept, removed


def thresholds_from_args(values: Iterable[float] | None) -> list[float]:
    if values is None:
        return [0.8]
    return sorted({float(value) for value in values})


def selected_thresholds(probability: float, thresholds: Iterable[float]) -> list[float]:
    return [float(threshold) for threshold in thresholds if probability >= float(threshold)]


def build_replay_target(
    prediction: dict[str, str],
    source_row: dict[str, Any],
    *,
    thresholds: list[float],
) -> dict[str, Any]:
    probability = safe_float(prediction.get("risk_probability"))
    missing = replay_missing_fields(source_row)
    target = dict(source_row)
    target.update(
        {
            "schema": "hu_turn2_stage8c_topk_confirm_unknown_replay_target_v1",
            "replay_target_reason": "preconfirm_gate_selected_unknown_delta",
            "risk_probability": probability,
            "risk_prediction_split": prediction.get("split", ""),
            "risk_selected_thresholds": selected_thresholds(probability, thresholds),
            "risk_selected_thresholds_csv": ",".join(
                f"{value:g}" for value in selected_thresholds(probability, thresholds)
            ),
            "risk_prediction_label": safe_int(prediction.get("label"), -1),
            "risk_target_group": prediction.get("risk_target_group", source_row.get("risk_target_group", "")),
            "realized_delta_observed": False,
            "observed_performance_claim": "No",
            "replay_ready": not missing,
            "replay_blocker": ",".join(missing),
            "production_p2_fixed": "No-Go",
            "teacher_50k": "No-Go",
            "t1_training": "No-Go",
            "note": (
                "This row was selected by a deployable pre-confirm risk head, but its would-fire "
                "whole-game delta is unobserved. Replay before using it as performance evidence."
            ),
        }
    )
    return target


def extract_targets(
    distillation_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, str]],
    *,
    thresholds: list[float],
    splits: set[str] | None,
    excluded_recommended_uses: set[str] | None = None,
    max_targets: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    index = load_distillation_index(distillation_rows)
    excluded_recommended_uses = excluded_recommended_uses or set()
    target_by_key: dict[tuple[str, str, str, str, str, str, str, str], dict[str, Any]] = {}
    summary_counter: Counter[str] = Counter()
    threshold_rows: list[dict[str, Any]] = []

    for threshold in thresholds:
        selected = 0
        observed = 0
        unknown = 0
        matched_unknown = 0
        replay_ready = 0
        missing_source = 0
        excluded_predictions = 0
        for prediction in prediction_rows:
            split = str(prediction.get("split", ""))
            if splits is not None and split not in splits:
                continue
            prediction_use = str(prediction.get("recommended_training_use") or prediction.get("target_recommended_use") or "")
            if prediction_use in excluded_recommended_uses:
                excluded_predictions += 1
                continue
            if safe_float(prediction.get("risk_probability")) < threshold:
                continue
            selected += 1
            if truthy(prediction.get("realized_delta_observed")):
                observed += 1
                continue
            unknown += 1
            source_row = index.get(row_key(prediction))
            if source_row is None:
                missing_source += 1
                continue
            matched_unknown += 1
            target = build_replay_target(prediction, source_row, thresholds=thresholds)
            if target["replay_ready"]:
                replay_ready += 1
            key = row_key(source_row)
            previous = target_by_key.get(key)
            if previous is None or safe_float(target.get("risk_probability")) > safe_float(previous.get("risk_probability")):
                target_by_key[key] = target
        threshold_rows.append(
            {
                "threshold": threshold,
                "splits": "all" if splits is None else ",".join(sorted(splits)),
                "selected_predictions": selected,
                "selected_observed_delta": observed,
                "selected_unknown_delta": unknown,
                "matched_unknown_delta": matched_unknown,
                "missing_distillation_row": missing_source,
                "matched_replay_ready": replay_ready,
                "excluded_predictions": excluded_predictions,
            }
        )
        summary_counter["selected_predictions"] += selected
        summary_counter["selected_unknown_delta"] += unknown
        summary_counter["matched_unknown_delta"] += matched_unknown

    targets = sorted(
        target_by_key.values(),
        key=lambda row: (
            -safe_float(row.get("risk_probability")),
            str(row.get("hand_seed", "")),
            str(row.get("seat", "")),
            str(row.get("action_signature", "")),
        ),
    )
    if max_targets > 0:
        targets = targets[:max_targets]

    breakdown = build_breakdown_rows(targets)
    return targets, threshold_rows, breakdown


def build_breakdown_rows(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, str, list[dict[str, Any]]]] = [("overall", "all", targets)]
    for field in ("seat", "candidate_source", "recommended_training_use", "no_override_reason", "risk_prediction_split"):
        for value in sorted({str(row.get(field, "")) for row in targets}):
            groups.append((field, value, [row for row in targets if str(row.get(field, "")) == value]))
    for field, value, subset in groups:
        if not subset:
            continue
        probs = [safe_float(row.get("risk_probability")) for row in subset]
        rows.append(
            {
                "group_field": field,
                "group_value": value,
                "rows": len(subset),
                "replay_ready_rows": sum(1 for row in subset if truthy(row.get("replay_ready"))),
                "risk_probability_mean": sum(probs) / len(probs),
                "risk_probability_min": min(probs),
                "risk_probability_max": max(probs),
                "observed_delta_rows": sum(1 for row in subset if truthy(row.get("realized_delta_observed"))),
            }
        )
    return rows


def write_summary(path: Path, targets: list[dict[str, Any]], threshold_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# HU T2 Stage8c TopK Confirm Unknown Replay Targets",
        "",
        "This artifact is replay prep only. It does not claim runtime performance.",
        "",
        f"- Replay targets: `{len(targets)}`",
        f"- Replay-ready targets: `{sum(1 for row in targets if truthy(row.get('replay_ready')))}`",
        f"- Observed realized deltas in targets: `{sum(1 for row in targets if truthy(row.get('realized_delta_observed')))}`",
        "- Production / P2 fixed / 50k / T1: `No-Go`",
        "",
        "## Thresholds",
        "",
        "| threshold | selected | observed | unknown | matched unknown | replay-ready | missing source | excluded predictions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in threshold_rows:
        lines.append(
            "| {threshold:g} | {selected_predictions} | {selected_observed_delta} | "
            "{selected_unknown_delta} | {matched_unknown_delta} | {matched_replay_ready} | "
            "{missing_distillation_row} | {excluded_predictions} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Rows here were selected by a pre-confirm deployable head, but their would-fire realized delta is unknown.",
            "- They must be replayed with an independent whole-game paired estimator before being used as gate evidence.",
            "- Rejected/topk_empty placeholder zero deltas remain excluded from performance claims.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    thresholds = thresholds_from_args(args.threshold)
    splits = {str(value) for value in args.split} if args.split else None
    source_counts: dict[str, int] = {}
    distillation_rows: list[dict[str, Any]] = []
    for path in args.distillation_jsonl:
        loaded = read_jsonl(path)
        source_counts[str(path)] = len(loaded)
        distillation_rows.extend(loaded)
    loaded_distillation_rows = len(distillation_rows)
    distillation_rows, excluded_counts = filter_distillation_rows(
        distillation_rows,
        excluded_recommended_uses=args.exclude_recommended_use,
    )
    prediction_rows = read_csv(args.predictions_csv)
    targets, threshold_rows, breakdown = extract_targets(
        distillation_rows,
        prediction_rows,
        thresholds=thresholds,
        splits=splits,
        excluded_recommended_uses={str(value) for value in args.exclude_recommended_use},
        max_targets=args.max_targets,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / DEFAULT_TARGET_NAME, targets)
    write_csv(args.output_dir / "topk_confirm_unknown_replay_target_thresholds.csv", threshold_rows)
    write_csv(args.output_dir / "topk_confirm_unknown_replay_target_breakdown.csv", breakdown)
    write_summary(args.output_dir / "topk_confirm_unknown_replay_target_summary.md", targets, threshold_rows)
    manifest = {
        "schema": "hu_turn2_stage8c_topk_confirm_unknown_replay_target_manifest_v1",
        "distillation_jsonl": [str(path) for path in args.distillation_jsonl],
        "predictions_csv": str(args.predictions_csv),
        "output_dir": str(args.output_dir),
        "source_counts": source_counts,
        "loaded_distillation_rows": loaded_distillation_rows,
        "distillation_rows_after_filter": len(distillation_rows),
        "exclude_recommended_use": [str(value) for value in args.exclude_recommended_use],
        "excluded_recommended_use_counts": dict(excluded_counts),
        "excluded_recommended_use_rows": int(sum(excluded_counts.values())),
        "thresholds": thresholds,
        "splits": "all" if splits is None else sorted(splits),
        "targets": len(targets),
        "replay_ready_targets": sum(1 for row in targets if truthy(row.get("replay_ready"))),
        "production_p2_fixed": "No-Go",
        "teacher_50k": "No-Go",
        "t1_training": "No-Go",
    }
    (args.output_dir / "topk_confirm_unknown_replay_target_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
