"""Evaluate Stage8c fire-head predictions with a local-negative veto head.

This is an analysis-only bridge between two training outputs:

- a TopK confirm fire head, where high ``risk_probability`` means "fire"
- a local EV negative head, where high ``risk_probability`` means "veto"

Rows are joined by state/action/baseline signatures. The intended first use is
the MC512 replay-labeled TopK unknown rows, not production runtime approval.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_FIRE_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)
DEFAULT_VETO_THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fire-predictions", type=Path, required=True)
    parser.add_argument("--veto-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fire-thresholds",
        default=",".join(str(value) for value in DEFAULT_FIRE_THRESHOLDS),
    )
    parser.add_argument(
        "--veto-thresholds",
        default=",".join(str(value) for value in DEFAULT_VETO_THRESHOLDS),
        help="Candidate is kept only when veto_probability is below this threshold.",
    )
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out:
        return default
    return out


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_thresholds(raw: str) -> list[float]:
    values = sorted({float(part.strip()) for part in raw.split(",") if part.strip()})
    if not values:
        raise ValueError("at least one threshold is required")
    return values


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
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
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def join_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("state_signature") or ""),
        str(row.get("action_signature") or ""),
        str(row.get("baseline_action_signature") or ""),
    )


def join_predictions(
    fire_rows: list[dict[str, Any]],
    veto_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    veto_by_key = {join_key(row): row for row in veto_rows}
    joined: list[dict[str, Any]] = []
    duplicate_fire_keys = 0
    seen_fire_keys: set[tuple[str, str, str]] = set()
    for fire in fire_rows:
        key = join_key(fire)
        if key in seen_fire_keys:
            duplicate_fire_keys += 1
        seen_fire_keys.add(key)
        veto = veto_by_key.get(key)
        if veto is None:
            continue
        joined.append(
            {
                "state_signature": key[0],
                "action_signature": key[1],
                "baseline_action_signature": key[2],
                "source_log": fire.get("source_log", ""),
                "hand_seed": fire.get("hand_seed", ""),
                "seat": fire.get("seat", ""),
                "candidate_source": fire.get("candidate_source", ""),
                "fire_split": fire.get("split", ""),
                "veto_split": veto.get("split", ""),
                "fire_probability": safe_float(fire.get("risk_probability")),
                "veto_probability": safe_float(veto.get("risk_probability")),
                "fire_label": safe_int(fire.get("label")),
                "veto_label": safe_int(veto.get("label")),
                "recommended_training_use": fire.get("recommended_training_use", ""),
                "local_replay_label": fire.get("local_replay_label", veto.get("local_replay_label", "")),
                "local_replay_bucket": fire.get("local_replay_bucket", veto.get("local_replay_bucket", "")),
                "realized_delta": safe_float(fire.get("realized_delta", veto.get("realized_delta"))),
                "realized_delta_observed": safe_int(
                    fire.get("realized_delta_observed", veto.get("realized_delta_observed"))
                ),
                "local_replay_delta": safe_float(fire.get("local_replay_delta", veto.get("local_replay_delta"))),
            }
        )
    return joined, {
        "fire_rows": len(fire_rows),
        "veto_rows": len(veto_rows),
        "joined_rows": len(joined),
        "fire_rows_without_veto": len(fire_rows) - len(joined),
        "duplicate_fire_keys": duplicate_fire_keys,
    }


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def metric_row(rows: list[dict[str, Any]], *, fire_threshold: float, veto_threshold: float) -> dict[str, Any]:
    fire_selected = [row for row in rows if safe_float(row["fire_probability"]) >= fire_threshold]
    kept = [row for row in fire_selected if safe_float(row["veto_probability"]) < veto_threshold]
    vetoed = [row for row in fire_selected if safe_float(row["veto_probability"]) >= veto_threshold]
    positives = [row for row in kept if safe_int(row["fire_label"]) == 1]
    negatives = [row for row in kept if safe_int(row["fire_label"]) == 0]
    vetoed_negatives = [row for row in vetoed if safe_int(row["fire_label"]) == 0]
    vetoed_positives = [row for row in vetoed if safe_int(row["fire_label"]) == 1]
    kept_deltas = [safe_float(row["local_replay_delta"] or row["realized_delta"]) for row in kept]
    fire_deltas = [safe_float(row["local_replay_delta"] or row["realized_delta"]) for row in fire_selected]
    return {
        "fire_threshold": fire_threshold,
        "veto_threshold": veto_threshold,
        "rows": len(rows),
        "fire_selected": len(fire_selected),
        "kept_after_veto": len(kept),
        "vetoed": len(vetoed),
        "kept_positive": len(positives),
        "kept_negative": len(negatives),
        "kept_precision": len(positives) / max(len(kept), 1),
        "kept_negative_rate": len(negatives) / max(len(kept), 1),
        "kept_delta_mean": mean(kept_deltas),
        "kept_loss_count": sum(1 for value in kept_deltas if value < 0.0),
        "kept_max_loss": max((max(0.0, -value) for value in kept_deltas), default=0.0),
        "fire_only_delta_mean": mean(fire_deltas),
        "vetoed_negative": len(vetoed_negatives),
        "vetoed_positive": len(vetoed_positives),
        "veto_precision_for_negative": len(vetoed_negatives) / max(len(vetoed), 1),
        "veto_negative_recall_within_fire": len(vetoed_negatives)
        / max(sum(1 for row in fire_selected if safe_int(row["fire_label"]) == 0), 1),
        "veto_positive_loss_within_fire": len(vetoed_positives)
        / max(sum(1 for row in fire_selected if safe_int(row["fire_label"]) == 1), 1),
    }


def split_metric_rows(
    rows: list[dict[str, Any]],
    *,
    fire_thresholds: list[float],
    veto_thresholds: list[float],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    split_values = sorted({str(row.get("fire_split") or "unknown") for row in rows})
    split_groups = [("all", rows)] + [
        (split, [row for row in rows if str(row.get("fire_split") or "unknown") == split]) for split in split_values
    ]
    for split, split_rows in split_groups:
        for fire_threshold in fire_thresholds:
            for veto_threshold in veto_thresholds:
                out.append(
                    {"split": split}
                    | metric_row(split_rows, fire_threshold=fire_threshold, veto_threshold=veto_threshold)
                )
    return out


def write_summary(path: Path, summary: dict[str, Any], best_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# HU T2 Stage8c Fire + Veto Combo Audit",
        "",
        "This is analysis-only. It does not approve production, P2 fixed status, T1, or 50k teacher generation.",
        "",
        "## Join",
        "",
        f"- fire rows: `{summary['fire_rows']}`",
        f"- veto rows: `{summary['veto_rows']}`",
        f"- joined rows: `{summary['joined_rows']}`",
        "",
        "## Best All-Split Rows",
        "",
        "| fire | veto | kept | precision | neg rate | mean delta | veto neg | veto pos |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_rows[:10]:
        lines.append(
            "| {fire:.2f} | {veto:.2f} | {kept} | {precision:.3f} | {neg_rate:.3f} | {delta:.3f} | {vn} | {vp} |".format(
                fire=float(row["fire_threshold"]),
                veto=float(row["veto_threshold"]),
                kept=int(row["kept_after_veto"]),
                precision=float(row["kept_precision"]),
                neg_rate=float(row["kept_negative_rate"]),
                delta=float(row["kept_delta_mean"]),
                vn=int(row["vetoed_negative"]),
                vp=int(row["vetoed_positive"]),
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- combo audit: `diagnostic only`",
            "- production / P2 fixed: `No-Go`",
            "- 50k teacher: `No-Go`",
            "- T1 training: `No-Go`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    fire_rows = read_csv(args.fire_predictions)
    veto_rows = read_csv(args.veto_predictions)
    joined, join_summary = join_predictions(fire_rows, veto_rows)
    metric_rows = split_metric_rows(
        joined,
        fire_thresholds=parse_thresholds(args.fire_thresholds),
        veto_thresholds=parse_thresholds(args.veto_thresholds),
    )
    all_rows = [row for row in metric_rows if row["split"] == "all" and int(row["kept_after_veto"]) > 0]
    all_rows.sort(
        key=lambda row: (
            float(row["kept_delta_mean"]),
            float(row["kept_precision"]),
            int(row["kept_after_veto"]),
        ),
        reverse=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "fire_veto_joined_rows.csv", joined)
    write_csv(args.output_dir / "fire_veto_combo_metrics.csv", metric_rows)
    write_csv(args.output_dir / "fire_veto_combo_best_all.csv", all_rows[:50])
    write_jsonl(args.output_dir / "fire_veto_joined_rows.jsonl", joined)
    (args.output_dir / "fire_veto_join_summary.json").write_text(
        json.dumps(join_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary(args.output_dir / "fire_veto_combo_summary.md", join_summary, all_rows)
    print(json.dumps(join_summary, sort_keys=True))


if __name__ == "__main__":
    main()
