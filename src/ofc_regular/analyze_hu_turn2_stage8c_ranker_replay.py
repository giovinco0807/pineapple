"""Analyze independent replay results for Stage8c ranker-selected targets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT_DIR = Path("outputs/evals/hu_turn2_stage8c_ranker_replay_analysis")
PRIMARY_METRIC_SOURCE = "independent_replay_delta_for_label"
CONFIRM_DELTA_METRIC_ROLE = "source_gate_diagnostic_only"
CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-summary-csv", type=Path, required=True)
    parser.add_argument("--replay-teacher-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-loss-count", type=int, default=30)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def summary_by_row_index(summary_rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    return {safe_int(row.get("row_index"), -1): row for row in summary_rows}


def enriched_rows(summary_rows: list[dict[str, str]], teacher_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = summary_by_row_index(summary_rows)
    output: list[dict[str, Any]] = []
    for index, teacher in enumerate(teacher_rows):
        source = teacher.get("topk_hard_negative_source", {})
        replay = teacher.get("topk_hard_negative_replay", {})
        summary = summaries.get(index, {})
        row = {
            "row_index": index,
            "status": summary.get("status", ""),
            "action_mapping_status": replay.get("action_mapping_status", summary.get("action_mapping_status", "")),
            "ranker": source.get("ranker", ""),
            "ranker_rank": safe_int(source.get("ranker_rank"), -1),
            "ranker_score": safe_float(source.get("ranker_score")),
            "hand_seed": teacher.get("hand_seed", source.get("hand_seed", "")),
            "seat": source.get("seat", teacher.get("seat", "")),
            "candidate_source": source.get("candidate_source", ""),
            "recommended_training_use": source.get("recommended_training_use", ""),
            "candidate_ev_rank": safe_float(source.get("candidate_ev_rank"), 9999.0),
            "fire_probability": safe_float(source.get("fire_probability")),
            "policy_delta_prediction": safe_float(source.get("policy_delta_prediction")),
            "predicted_delta": safe_float(source.get("predicted_delta")),
            "confirm_delta": safe_float(source.get("confirm_delta")),
            "confirm_delta_se": safe_float(source.get("confirm_delta_se")),
            "delta_for_label": safe_float(replay.get("delta_for_label", summary.get("delta_for_label"))),
            "delta_standard_error_for_label": safe_float(
                replay.get("delta_standard_error_for_label", summary.get("delta_standard_error_for_label"))
            ),
            "safe_lcb196_label": replay.get("safe_lcb196_label", summary.get("safe_lcb196_label", "")),
            "safe_lcb164_label": replay.get("safe_lcb164_label", summary.get("safe_lcb164_label", "")),
            "hard_negative_label": safe_int(replay.get("hard_negative_label", summary.get("hard_negative_label"))),
            "replay_delta_lcb196": safe_float(replay.get("replay_delta_lcb196", summary.get("replay_delta_lcb196"))),
            "replay_delta_lcb164": safe_float(replay.get("replay_delta_lcb164", summary.get("replay_delta_lcb164"))),
        }
        output.append(row)
    return output


def ok_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("status") == "ok" and row.get("action_mapping_status") == "ok"]


def aggregate(rows: list[dict[str, Any]], *, group_field: str, group_value: str) -> dict[str, Any]:
    ok = ok_rows(rows)
    deltas = [safe_float(row.get("delta_for_label")) for row in ok]
    losses = [max(0.0, -delta) for delta in deltas]
    std = sample_std(deltas)
    se_mean = std / math.sqrt(len(deltas)) if deltas else 0.0
    avg = mean(deltas)
    return {
        "group_field": group_field,
        "group_value": group_value,
        "rows": len(rows),
        "ok_rows": len(ok),
        "mapping_bad_rows": len(rows) - len(ok),
        "mean_delta": avg,
        "delta_ci95_low": avg - 1.96 * se_mean if deltas else 0.0,
        "delta_ci95_high": avg + 1.96 * se_mean if deltas else 0.0,
        "median_delta": percentile(deltas, 0.50),
        "p05_delta": percentile(deltas, 0.05),
        "p25_delta": percentile(deltas, 0.25),
        "p75_delta": percentile(deltas, 0.75),
        "p95_delta": percentile(deltas, 0.95),
        "min_delta": min(deltas) if deltas else 0.0,
        "max_delta": max(deltas) if deltas else 0.0,
        "negative_rows": sum(1 for value in deltas if value < 0.0),
        "negative_rate": sum(1 for value in deltas if value < 0.0) / len(deltas) if deltas else 0.0,
        "positive_rows": sum(1 for value in deltas if value > 0.0),
        "positive_rate": sum(1 for value in deltas if value > 0.0) / len(deltas) if deltas else 0.0,
        "hard_negative_rows": sum(safe_int(row.get("hard_negative_label")) for row in ok),
        "safe_lcb196_positive_rows": sum(1 for row in ok if str(row.get("safe_lcb196_label")) == "positive"),
        "safe_lcb196_gray_rows": sum(1 for row in ok if str(row.get("safe_lcb196_label")) == "gray"),
        "safe_lcb196_negative_rows": sum(1 for row in ok if str(row.get("safe_lcb196_label")) == "negative"),
        "safe_lcb164_positive_rows": sum(1 for row in ok if str(row.get("safe_lcb164_label")) == "positive"),
        "mean_delta_se": mean([safe_float(row.get("delta_standard_error_for_label")) for row in ok]),
        "mean_confirm_delta": mean([safe_float(row.get("confirm_delta")) for row in ok]),
        "primary_metric_source": PRIMARY_METRIC_SOURCE,
        "confirm_delta_metric_role": CONFIRM_DELTA_METRIC_ROLE,
        "confirm_delta_performance_claim_allowed": CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED,
        "mean_fire_probability": mean([safe_float(row.get("fire_probability")) for row in ok]),
        "mean_policy_delta_prediction": mean([safe_float(row.get("policy_delta_prediction")) for row in ok]),
        "p95_loss": percentile(losses, 0.95),
        "max_loss": max(losses) if losses else 0.0,
    }


def breakdown(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = [aggregate(rows, group_field="overall", group_value="all")]
    for field in ("ranker", "seat", "candidate_source", "recommended_training_use", "safe_lcb196_label"):
        for value in sorted({str(row.get(field, "")) for row in rows}):
            output.append(aggregate([row for row in rows if str(row.get(field, "")) == value], group_field=field, group_value=value))
    return output


def top_losses(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    ordered = sorted(ok_rows(rows), key=lambda row: safe_float(row.get("delta_for_label")))
    return ordered[:limit]


def write_summary(path: Path, breakdown_rows: list[dict[str, Any]]) -> None:
    overall = breakdown_rows[0] if breakdown_rows else {}
    rankers = [row for row in breakdown_rows if row.get("group_field") == "ranker"]
    rankers.sort(key=lambda row: safe_float(row.get("mean_delta")), reverse=True)
    lines = [
        "# HU T2 Stage8c Ranker Replay Analysis",
        "",
        "This is independent replay analysis for target selection only. It does not approve runtime, P2, production, T1, or 50k teacher generation.",
        f"Primary metric source is `{PRIMARY_METRIC_SOURCE}`. `mean_confirm_delta` is `{CONFIRM_DELTA_METRIC_ROLE}` and is not performance evidence.",
        "",
        f"- rows: `{overall.get('rows', 0)}`",
        f"- ok rows: `{overall.get('ok_rows', 0)}`",
        f"- overall mean delta: `{safe_float(overall.get('mean_delta')):+.4f}`",
        f"- overall 95% CI: `[{safe_float(overall.get('delta_ci95_low')):+.4f}, {safe_float(overall.get('delta_ci95_high')):+.4f}]`",
        f"- hard negatives: `{safe_int(overall.get('hard_negative_rows'))}`",
        f"- safe LCB196 positive rows: `{safe_int(overall.get('safe_lcb196_positive_rows'))}`",
        "",
        "## Rankers",
        "",
        "| ranker | rows | mean delta | CI low | CI high | neg rate | hard neg | LCB196+ | max loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rankers:
        lines.append(
            "| {group_value} | {ok_rows} | {mean_delta:+.4f} | {delta_ci95_low:+.4f} | {delta_ci95_high:+.4f} | {negative_rate:.3f} | {hard_negative_rows} | {safe_lcb196_positive_rows} | {max_loss:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- execution: `Pass`",
            "- allowed role: `MC512 labels for replay-target triage and hard-negative mining`",
            "- runtime gate: `No-Go`",
            "- production / P2 fixed: `No-Go`",
            "- 50k teacher: `No-Go`",
            "- T1 training: `No-Go`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(*, replay_summary_csv: Path, replay_teacher_jsonl: Path, output_dir: Path, top_loss_count: int) -> dict[str, Any]:
    rows = enriched_rows(read_csv(replay_summary_csv), read_jsonl(replay_teacher_jsonl))
    breakdown_rows = breakdown(rows)
    losses = top_losses(rows, limit=top_loss_count)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "ranker_replay_enriched_rows.csv", rows)
    write_csv(output_dir / "ranker_replay_breakdown.csv", breakdown_rows)
    write_jsonl(output_dir / "ranker_replay_top_losses.jsonl", losses)
    write_summary(output_dir / "ranker_replay_summary.md", breakdown_rows)
    manifest = {
        "replay_summary_csv": str(replay_summary_csv),
        "replay_teacher_jsonl": str(replay_teacher_jsonl),
        "rows": len(rows),
        "ok_rows": len(ok_rows(rows)),
        "output_dir": str(output_dir),
        "primary_metric_source": PRIMARY_METRIC_SOURCE,
        "confirm_delta_metric_role": CONFIRM_DELTA_METRIC_ROLE,
        "confirm_delta_performance_claim_allowed": CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED,
        "production_p2_fixed": "No-Go",
        "teacher_50k": "No-Go",
        "t1_training": "No-Go",
    }
    (output_dir / "ranker_replay_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = analyze(
        replay_summary_csv=args.replay_summary_csv,
        replay_teacher_jsonl=args.replay_teacher_jsonl,
        output_dir=args.output_dir,
        top_loss_count=args.top_loss_count,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
