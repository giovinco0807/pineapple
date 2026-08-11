"""Compare HU T2 Stage8c TopK-confirm fire-head training runs.

This is an analysis helper for research artifacts. It compares already-trained
``train_hu_turn2_stage8c_risk_head`` output directories and keeps the decision
separate from runtime approval: good ranking smoke results can justify more
labeling or candidate triage, but not production/P2/T1/50k by themselves.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


METRICS_FILE = "risk_head_metrics.csv"
THRESHOLD_FILE = "risk_head_threshold_metrics.csv"
TOPK_FILE = "risk_head_topk_metrics.csv"
MANIFEST_FILE = "risk_head_training_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Training output directory in label=path form. Repeat to compare multiple runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/training/hu_turn2_stage8c_fire_head_training_comparison"),
    )
    parser.add_argument(
        "--preferred-topk",
        type=int,
        default=3,
        help="TopK count used as the primary triage ranking tie-breaker.",
    )
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out:
        return default
    if out in (float("inf"), float("-inf")):
        return default
    return out


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_run_specs(specs: Iterable[str]) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--run must be label=path: {spec}")
        label, raw_path = spec.split("=", 1)
        label = label.strip()
        path = Path(raw_path.strip())
        if not label:
            raise SystemExit(f"--run label is empty: {spec}")
        if not path.is_dir():
            raise SystemExit(f"--run path is not a directory: {path}")
        runs.append((label, path))
    if len(runs) < 2:
        raise SystemExit("at least two --run entries are required")
    labels = [label for label, _path in runs]
    if len(labels) != len(set(labels)):
        raise SystemExit("--run labels must be unique")
    return runs


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def read_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / MANIFEST_FILE
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def by_split(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("split") or ""): row for row in rows}


def topk_row(rows: list[dict[str, Any]], split: str, topk: int) -> dict[str, Any]:
    for row in rows:
        if str(row.get("split") or "") == split and safe_int(row.get("topk")) == topk:
            return row
    return {}


def threshold_row(rows: list[dict[str, Any]], split: str, threshold: float) -> dict[str, Any]:
    for row in rows:
        if str(row.get("split") or "") == split and abs(safe_float(row.get("threshold")) - threshold) < 1e-9:
            return row
    return {}


def run_summary_row(label: str, path: Path, *, preferred_topk: int) -> dict[str, Any]:
    manifest = read_manifest(path)
    metrics = by_split(read_csv(path / METRICS_FILE))
    topk_rows = read_csv(path / TOPK_FILE)
    threshold_rows = read_csv(path / THRESHOLD_FILE)
    val_metrics = metrics.get("val", {})
    test_metrics = metrics.get("test", {})
    all_metrics = metrics.get("all", {})
    val_topk = topk_row(topk_rows, "val", preferred_topk)
    test_topk = topk_row(topk_rows, "test", preferred_topk)
    test_top3 = topk_row(topk_rows, "test", 3)
    test_top5 = topk_row(topk_rows, "test", 5)
    test_top10 = topk_row(topk_rows, "test", 10)
    test_threshold_08 = threshold_row(threshold_rows, "test", 0.8)
    test_threshold_09 = threshold_row(threshold_rows, "test", 0.9)
    test_threshold_05 = threshold_row(threshold_rows, "test", 0.5)
    excluded_counts = manifest.get("excluded_recommended_use_counts") or {}
    target_counts = manifest.get("target_group_counts") or {}
    runtime_gate_decision = "No-Go"
    if (
        safe_int(test_threshold_08.get("selected_rows")) >= 20
        and safe_float(test_threshold_08.get("precision")) >= 0.50
        and safe_float(test_threshold_08.get("selected_observed_realized_delta_mean")) > 0.0
    ):
        runtime_gate_decision = "Needs seat-swap validation"
    triage_score = (
        safe_float(test_metrics.get("roc_auc"))
        + safe_float(test_metrics.get("average_precision"))
        + safe_float(test_topk.get("precision"))
        + 0.25 * safe_float(test_top10.get("precision"))
    )
    return {
        "run": label,
        "path": str(path),
        "trainable_rows": safe_int(manifest.get("trainable_rows")),
        "positive_rows": safe_int(manifest.get("positive_rows")),
        "negative_rows": safe_int(manifest.get("negative_rows")),
        "excluded_topk_empty_rows": safe_int(excluded_counts.get("topk_confirm_topk_empty")),
        "realized_positive_rows": safe_int(target_counts.get("topk_confirm_realized_positive")),
        "realized_loss_rows": safe_int(target_counts.get("topk_confirm_realized_loss")),
        "confirm_rejected_rows": safe_int(target_counts.get("topk_confirm_rejected")),
        "replay_positive_rows": safe_int(target_counts.get("topk_confirm_replay_positive")),
        "replay_negative_rows": safe_int(target_counts.get("topk_confirm_replay_negative")),
        "replay_gray_rows": safe_int(target_counts.get("topk_confirm_replay_gray")),
        "pos_weight_mode": str(manifest.get("pos_weight_mode") or ""),
        "pos_weight": safe_float(manifest.get("pos_weight")),
        "topk_realized_loss_weight": safe_float(manifest.get("topk_realized_loss_weight"), 1.0),
        "feature_mode": str(manifest.get("feature_mode") or ""),
        "split_mode": str(manifest.get("split_mode") or ""),
        "val_ap": safe_float(val_metrics.get("average_precision")),
        "test_ap": safe_float(test_metrics.get("average_precision")),
        "all_ap": safe_float(all_metrics.get("average_precision")),
        "val_roc_auc": safe_float(val_metrics.get("roc_auc")),
        "test_roc_auc": safe_float(test_metrics.get("roc_auc")),
        "all_roc_auc": safe_float(all_metrics.get("roc_auc")),
        f"val_top{preferred_topk}_precision": safe_float(val_topk.get("precision")),
        f"test_top{preferred_topk}_precision": safe_float(test_topk.get("precision")),
        "test_top3_precision": safe_float(test_top3.get("precision")),
        "test_top5_precision": safe_float(test_top5.get("precision")),
        "test_top10_precision": safe_float(test_top10.get("precision")),
        f"test_top{preferred_topk}_observed_delta_mean": safe_float(
            test_topk.get("selected_observed_realized_delta_mean")
        ),
        "test_top5_observed_delta_mean": safe_float(test_top5.get("selected_observed_realized_delta_mean")),
        "test_top10_observed_delta_mean": safe_float(test_top10.get("selected_observed_realized_delta_mean")),
        "test_threshold_0p5_precision": safe_float(test_threshold_05.get("precision")),
        "test_threshold_0p5_selected_rows": safe_int(test_threshold_05.get("selected_rows")),
        "test_threshold_0p8_precision": safe_float(test_threshold_08.get("precision")),
        "test_threshold_0p8_selected_rows": safe_int(test_threshold_08.get("selected_rows")),
        "test_threshold_0p9_precision": safe_float(test_threshold_09.get("precision")),
        "test_threshold_0p9_selected_rows": safe_int(test_threshold_09.get("selected_rows")),
        "triage_score": triage_score,
        "runtime_gate_decision": runtime_gate_decision,
    }


def attach_run_label(label: str, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [({"run": label} | dict(row)) for row in rows]


def compare_runs(runs: list[tuple[str, Path]], *, preferred_topk: int) -> dict[str, Any]:
    summary_rows = [run_summary_row(label, path, preferred_topk=preferred_topk) for label, path in runs]
    summary_rows.sort(key=lambda row: safe_float(row.get("triage_score")), reverse=True)
    threshold_rows: list[dict[str, Any]] = []
    topk_rows: list[dict[str, Any]] = []
    for label, path in runs:
        threshold_rows.extend(attach_run_label(label, read_csv(path / THRESHOLD_FILE)))
        topk_rows.extend(attach_run_label(label, read_csv(path / TOPK_FILE)))
    preferred = summary_rows[0] if summary_rows else {}
    return {
        "summary_rows": summary_rows,
        "threshold_rows": threshold_rows,
        "topk_rows": topk_rows,
        "preferred_run": str(preferred.get("run") or ""),
        "preferred_role": "triage_only",
        "runtime_gate_decision": "No-Go",
    }


def write_markdown(path: Path, comparison: dict[str, Any]) -> None:
    summary_rows = comparison["summary_rows"]
    preferred = str(comparison.get("preferred_run") or "")
    lines = [
        "# HU T2 Stage8c Fire-Head Training Comparison",
        "",
        "This artifact compares training smokes only. It does not approve runtime use, P2 fixed status, T1, or 50k teacher generation.",
        "",
        "## Decision",
        "",
        f"- preferred triage run: `{preferred}`",
        "- runtime gate: `No-Go`",
        "- allowed role: `ranking/triage for selecting rows for expensive TopK+confirm or high-MC labeling`",
        "",
        "## Summary",
        "",
        "| run | rows | pos | replay +/-/gray | test AP | test AUC | test top3 precision | test top10 precision | th0.8 precision | th0.8 selected | decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            "| {run} | {rows} | {pos} | {rpos}/{rneg}/{rgray} | {ap:.4f} | {auc:.4f} | {top3:.3f} | {top10:.3f} | {thp:.3f} | {ths} | {decision} |".format(
                run=row["run"],
                rows=row["trainable_rows"],
                pos=row["positive_rows"],
                rpos=row["replay_positive_rows"],
                rneg=row["replay_negative_rows"],
                rgray=row["replay_gray_rows"],
                ap=safe_float(row["test_ap"]),
                auc=safe_float(row["test_roc_auc"]),
                top3=safe_float(row.get("test_top3_precision")),
                top10=safe_float(row["test_top10_precision"]),
                thp=safe_float(row["test_threshold_0p8_precision"]),
                ths=row["test_threshold_0p8_selected_rows"],
                decision=row["runtime_gate_decision"],
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `topk_confirm_topk_empty` controls should not dominate this comparison; they are easy no-op negatives.",
            "- Fixed threshold precision remains too low for runtime gating.",
            "- Use realized whole-game per-fire evidence, not confirm-gate means, for adoption-quality claims.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs = parse_run_specs(args.run)
    comparison = compare_runs(runs, preferred_topk=args.preferred_topk)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "fire_head_training_comparison.csv", comparison["summary_rows"])
    write_csv(args.output_dir / "fire_head_threshold_comparison.csv", comparison["threshold_rows"])
    write_csv(args.output_dir / "fire_head_topk_comparison.csv", comparison["topk_rows"])
    (args.output_dir / "fire_head_training_comparison_manifest.json").write_text(
        json.dumps(
            {
                "schema": "hu_turn2_stage8c_fire_head_training_comparison_v1",
                "preferred_run": comparison["preferred_run"],
                "preferred_role": comparison["preferred_role"],
                "runtime_gate_decision": comparison["runtime_gate_decision"],
                "run_count": len(runs),
                "preferred_topk": int(args.preferred_topk),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_markdown(args.output_dir / "fire_head_training_comparison.md", comparison)
    print(json.dumps({"preferred_run": comparison["preferred_run"], "runtime_gate_decision": "No-Go"}, sort_keys=True))


if __name__ == "__main__":
    main()
