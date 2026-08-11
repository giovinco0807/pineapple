"""Analyze Stage8c fire+veto replay targets with independent replay deltas.

This is an evaluation-only diagnostic. Fire and veto probabilities are model
diagnostics; the performance signal is the independently replayed delta.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT_DIR = Path("outputs/evals/hu_turn2_stage8c_fire_veto_replay_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-summary-csv", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-loss-count", type=int, default=30)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


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


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((value - m) ** 2 for value in values) / (len(values) - 1))


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def probability_bin(value: Any, cuts: tuple[float, ...]) -> str:
    x = safe_float(value)
    low = 0.0
    for cut in cuts:
        if x < cut:
            return f"[{low:.2f},{cut:.2f})"
        low = cut
    return f">={cuts[-1]:.2f}"


def source_for_summary_row(summary_row: dict[str, str], source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    index = safe_int(summary_row.get("row_index"), -1)
    if 0 <= index < len(source_rows):
        return source_rows[index]
    return {}


def reason_members(source: dict[str, Any]) -> list[str]:
    raw = str(source.get("replay_target_reasons") or "")
    reasons = [part.strip() for part in raw.split(",") if part.strip()]
    if reasons:
        return reasons
    fallback = str(source.get("replay_target_reason") or "").strip()
    return [fallback] if fallback else ["unknown"]


def enrich_rows(summary_rows: list[dict[str, str]], source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for summary in summary_rows:
        source = source_for_summary_row(summary, source_rows)
        fire_probability = safe_float(source.get("fire_probability"))
        veto_probability = safe_float(source.get("veto_probability"))
        row = dict(summary)
        row.update(
            {
                "candidate_source": source.get("candidate_source", ""),
                "no_override_reason": source.get("no_override_reason", ""),
                "recommended_training_use": source.get("recommended_training_use", ""),
                "replay_target_reason": source.get("replay_target_reason", ""),
                "replay_target_reasons": ",".join(reason_members(source)),
                "fire_probability": fire_probability,
                "veto_probability": veto_probability,
                "fire_probability_bin": probability_bin(fire_probability, (0.80, 0.85, 0.90, 0.95)),
                "veto_probability_bin": probability_bin(veto_probability, (0.30, 0.40, 0.50, 0.60)),
                "fire_prediction_label": safe_int(source.get("fire_prediction_label"), -1),
                "veto_prediction_label": safe_int(source.get("veto_prediction_label"), -1),
                "veto_prediction_missing": truthy(source.get("veto_prediction_missing")),
                "source_replay_ready": truthy(source.get("replay_ready")),
                "source_replay_blocker": source.get("replay_blocker", ""),
            }
        )
        enriched.append(row)
    return enriched


def ok_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("status") == "ok" and row.get("action_mapping_status") == "ok"]


def aggregate_group(rows: list[dict[str, Any]], *, group_field: str, group_value: str) -> dict[str, Any]:
    ok = ok_rows(rows)
    deltas = [safe_float(row.get("delta_for_label")) for row in ok]
    losses = [max(0.0, -delta) for delta in deltas]
    std = sample_std(deltas)
    se_mean = std / math.sqrt(len(deltas)) if deltas else 0.0
    delta_mean = mean(deltas)
    return {
        "group_field": group_field,
        "group_value": group_value,
        "rows": len(rows),
        "ok_rows": len(ok),
        "mapping_bad_rows": len(rows) - len(ok),
        "mean_delta": delta_mean,
        "delta_ci95_low": delta_mean - 1.96 * se_mean if deltas else 0.0,
        "delta_ci95_high": delta_mean + 1.96 * se_mean if deltas else 0.0,
        "median_delta": percentile(deltas, 0.50),
        "p05_delta": percentile(deltas, 0.05),
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
        "mean_fire_probability": mean([safe_float(row.get("fire_probability")) for row in ok]),
        "mean_veto_probability": mean([safe_float(row.get("veto_probability")) for row in ok]),
        "mean_delta_se": mean([safe_float(row.get("delta_standard_error_for_label")) for row in ok]),
        "mean_loss": mean(losses),
        "p95_loss": percentile(losses, 0.95),
        "max_loss": max(losses) if losses else 0.0,
    }


def breakdown_by_field(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    return [
        aggregate_group([row for row in rows if str(row.get(field, "")) == value], group_field=field, group_value=value)
        for value in sorted({str(row.get(field, "")) for row in rows})
    ]


def reason_breakdown(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_reason: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for reason in [part.strip() for part in str(row.get("replay_target_reasons", "")).split(",") if part.strip()]:
            by_reason.setdefault(reason, []).append(row)
    return [
        aggregate_group(reason_rows, group_field="replay_target_reason_member", group_value=reason)
        for reason, reason_rows in sorted(by_reason.items())
    ]


def top_losses(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    ordered = sorted(ok_rows(rows), key=lambda row: safe_float(row.get("delta_for_label")))
    output: list[dict[str, Any]] = []
    for row in ordered[:limit]:
        output.append(
            {
                "row_index": row.get("row_index"),
                "hand_seed": row.get("hand_seed"),
                "seat": row.get("seat"),
                "candidate_source": row.get("candidate_source", ""),
                "no_override_reason": row.get("no_override_reason", ""),
                "replay_target_reasons": row.get("replay_target_reasons", ""),
                "fire_probability": safe_float(row.get("fire_probability")),
                "veto_probability": safe_float(row.get("veto_probability")),
                "fire_probability_bin": row.get("fire_probability_bin", ""),
                "veto_probability_bin": row.get("veto_probability_bin", ""),
                "delta_for_label": safe_float(row.get("delta_for_label")),
                "delta_standard_error_for_label": safe_float(row.get("delta_standard_error_for_label")),
                "safe_lcb196_label": row.get("safe_lcb196_label"),
                "hard_negative_label": safe_int(row.get("hard_negative_label")),
                "old_predicted_delta": safe_float(row.get("old_predicted_delta")),
                "old_gate_probability": safe_float(row.get("old_gate_probability")),
                "source_log": row.get("source_log", ""),
            }
        )
    return output


def decision(overall: dict[str, Any]) -> str:
    if safe_int(overall.get("mapping_bad_rows")) > 0:
        return "No-Go: action mapping failures"
    if safe_float(overall.get("delta_ci95_low")) <= 0.0:
        return "No-Go: replay CI low is not positive"
    if safe_float(overall.get("negative_rate")) > 0.10:
        return "No-Go: negative replay rate is high"
    return "Continue: positive replay signal, still not production"


def analyze(
    *,
    replay_summary_rows: list[dict[str, str]],
    source_rows: list[dict[str, Any]],
    top_loss_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    enriched = enrich_rows(replay_summary_rows, source_rows)
    breakdown: list[dict[str, Any]] = [aggregate_group(enriched, group_field="overall", group_value="all")]
    for field in (
        "seat",
        "candidate_source",
        "no_override_reason",
        "recommended_training_use",
        "safe_lcb196_label",
        "fire_probability_bin",
        "veto_probability_bin",
        "fire_prediction_label",
        "veto_prediction_label",
    ):
        breakdown.extend(breakdown_by_field(enriched, field))
    reasons = reason_breakdown(enriched)
    losses = top_losses(enriched, limit=top_loss_count)
    return enriched, breakdown, reasons, losses


def write_summary(
    path: Path,
    *,
    breakdown: list[dict[str, Any]],
    reason_rows: list[dict[str, Any]],
    loss_rows: list[dict[str, Any]],
) -> None:
    overall = breakdown[0] if breakdown else {}
    lines = [
        "# HU T2 Stage8c Fire + Veto Replay Analysis",
        "",
        "- Purpose: diagnose fire+veto replay targets with independent replay deltas.",
        "- This is not production/P2/50k/T1 approval.",
        f"- Rows: `{overall.get('rows', 0)}`",
        f"- OK rows: `{overall.get('ok_rows', 0)}`",
        f"- Mean delta: `{safe_float(overall.get('mean_delta')):+.4f}`",
        f"- 95% CI: `[{safe_float(overall.get('delta_ci95_low')):+.4f}, {safe_float(overall.get('delta_ci95_high')):+.4f}]`",
        f"- Negative rate: `{safe_float(overall.get('negative_rate')):.3f}`",
        f"- Hard negatives: `{safe_int(overall.get('hard_negative_rows'))}`",
        f"- Decision: `{decision(overall)}`",
        "",
        "## Reason Members",
        "",
        "| reason | rows | mean delta | CI low | CI high | negative rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in reason_rows:
        lines.append(
            "| {reason} | {rows} | {mean:+.4f} | {low:+.4f} | {high:+.4f} | {neg:.1%} |".format(
                reason=row["group_value"],
                rows=row["rows"],
                mean=safe_float(row.get("mean_delta")),
                low=safe_float(row.get("delta_ci95_low")),
                high=safe_float(row.get("delta_ci95_high")),
                neg=safe_float(row.get("negative_rate")),
            )
        )
    lines.extend(
        [
            "",
            "## Top Losses",
            "",
            "| row | seat | source | fire | veto | delta | label |",
            "|---:|---|---|---:|---:|---:|---|",
        ]
    )
    for row in loss_rows[:10]:
        lines.append(
            "| {row_index} | {seat} | {source} | {fire:.3f} | {veto:.3f} | {delta:+.4f} | {label} |".format(
                row_index=row.get("row_index"),
                seat=row.get("seat"),
                source=row.get("candidate_source", ""),
                fire=safe_float(row.get("fire_probability")),
                veto=safe_float(row.get("veto_probability")),
                delta=safe_float(row.get("delta_for_label")),
                label=row.get("safe_lcb196_label"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Fire/veto probabilities are diagnostics only.",
            "- The performance claim must come from replay delta or realized seat-swap delta.",
            "- Production / P2 fixed / T1 / 50k teacher remain `No-Go` unless a later replay set clears the decision gates.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    replay_summary_rows = read_csv(args.replay_summary_csv)
    source_rows = read_jsonl(args.source_jsonl)
    enriched, breakdown, reasons, losses = analyze(
        replay_summary_rows=replay_summary_rows,
        source_rows=source_rows,
        top_loss_count=args.top_loss_count,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "fire_veto_replay_enriched.csv", enriched)
    write_csv(args.output_dir / "fire_veto_group_breakdown.csv", breakdown)
    write_csv(args.output_dir / "fire_veto_reason_breakdown.csv", reasons)
    write_jsonl(args.output_dir / "fire_veto_top_losses.jsonl", losses)
    write_summary(args.output_dir / "fire_veto_replay_analysis.md", breakdown=breakdown, reason_rows=reasons, loss_rows=losses)
    overall = breakdown[0] if breakdown else {}
    print(
        json.dumps(
            {
                "schema": "hu_turn2_stage8c_fire_veto_replay_analysis_v1",
                "rows": safe_int(overall.get("rows")),
                "ok_rows": safe_int(overall.get("ok_rows")),
                "mean_delta": safe_float(overall.get("mean_delta")),
                "delta_ci95_low": safe_float(overall.get("delta_ci95_low")),
                "negative_rate": safe_float(overall.get("negative_rate")),
                "decision": decision(overall),
                "output_dir": str(args.output_dir),
                "production_p2_fixed": "No-Go",
                "teacher_50k": "No-Go",
                "t1_training": "No-Go",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
