"""Analyze HU T2 Stage8c fire-head runs with source-heldout as the unit.

This helper intentionally treats source-seed heldout performance as the
promotion gate. A selector that looks useful on one heldout source but loses on
another remains a research/triage artifact, not a runtime gate.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


PREDICTIONS_FILE = "risk_head_predictions.csv"
METRICS_FILE = "risk_head_metrics.csv"
MANIFEST_FILE = "risk_head_training_manifest.json"
DEFAULT_THRESHOLDS = (0.5, 0.7, 0.8, 0.9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Training output directory in label=path form. Repeat for each fixed-source run.",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        type=float,
        default=[],
        help="Risk-probability threshold to evaluate. Repeatable; defaults to 0.5/0.7/0.8/0.9.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/training/hu_turn2_stage8c_source_heldout_analysis"),
    )
    parser.add_argument(
        "--min-selected-per-source",
        type=int,
        default=20,
        help="Minimum selected rows per source for a threshold to be considered stable.",
    )
    parser.add_argument(
        "--min-sources-with-selection",
        type=int,
        default=3,
        help="Minimum heldout sources that must select enough rows for a threshold to be considered stable.",
    )
    parser.add_argument(
        "--require-all-sources-nonnegative",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require every source with enough selected rows to have nonnegative estimated delta/row.",
    )
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or out in (float("inf"), float("-inf")):
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
        if not (path / PREDICTIONS_FILE).exists():
            raise SystemExit(f"missing {PREDICTIONS_FILE}: {path}")
        runs.append((label, path))
    if not runs:
        raise SystemExit("at least one --run is required")
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
    seen: set[str] = set()
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


def test_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("split") or "") == "test"]


def row_source(row: dict[str, Any], manifest: dict[str, Any]) -> str:
    source = str(row.get("split_group_source_seed") or "").strip()
    if source:
        return source
    fixed = manifest.get("fixed_test_groups") or []
    if len(fixed) == 1:
        return str(fixed[0])
    raw = str(row.get("source_log") or row.get("split_group_source_log") or "").strip()
    return raw or "unknown"


def is_observed(row: dict[str, Any]) -> bool:
    raw = row.get("realized_delta_observed")
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return str(row.get("realized_delta") or "") != ""
    text = str(raw).strip().lower()
    return text in {"1", "true", "yes", "y"}


def source_test_metrics(label: str, path: Path) -> dict[str, Any]:
    manifest = read_manifest(path)
    metrics_by_split = {str(row.get("split") or ""): row for row in read_csv(path / METRICS_FILE)}
    predictions = test_rows(read_csv(path / PREDICTIONS_FILE))
    sources = sorted({row_source(row, manifest) for row in predictions})
    test_metrics = metrics_by_split.get("test", {})
    positives = sum(1 for row in predictions if safe_int(row.get("label")) == 1)
    observed = sum(1 for row in predictions if is_observed(row))
    return {
        "run": label,
        "path": str(path),
        "test_sources": ";".join(sources),
        "test_source_count": len(sources),
        "test_rows": len(predictions),
        "test_positive_rows": positives,
        "test_observed_rows": observed,
        "test_average_precision": safe_float(test_metrics.get("average_precision")),
        "test_roc_auc": safe_float(test_metrics.get("roc_auc")),
        "test_brier": safe_float(test_metrics.get("brier")),
        "fixed_test_groups": ";".join(str(item) for item in (manifest.get("fixed_test_groups") or [])),
        "split_mode": str(manifest.get("split_mode") or ""),
        "feature_mode": str(manifest.get("feature_mode") or ""),
    }


def threshold_metrics_for_source(
    *,
    label: str,
    path: Path,
    threshold: float,
    source: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = [row for row in rows if safe_float(row.get("risk_probability")) >= threshold]
    positives = sum(1 for row in rows if safe_int(row.get("label")) == 1)
    selected_positive = sum(1 for row in selected if safe_int(row.get("label")) == 1)
    selected_negative = len(selected) - selected_positive
    observed_selected = [row for row in selected if is_observed(row)]
    observed_delta_sum = sum(safe_float(row.get("realized_delta")) for row in observed_selected)
    observed_loss_values = [
        -safe_float(row.get("realized_delta"))
        for row in observed_selected
        if safe_float(row.get("realized_delta")) < 0.0
    ]
    selected_rows = len(selected)
    precision = selected_positive / selected_rows if selected_rows else 0.0
    recall = selected_positive / positives if positives else 0.0
    return {
        "run": label,
        "path": str(path),
        "source": source,
        "threshold": threshold,
        "rows": len(rows),
        "positive_rows": positives,
        "selected_rows": selected_rows,
        "selected_positive_rows": selected_positive,
        "selected_negative_rows": selected_negative,
        "precision": precision,
        "recall": recall,
        "fire_rate": selected_rows / len(rows) if rows else 0.0,
        "selected_observed_rows": len(observed_selected),
        "selected_unknown_rows": selected_rows - len(observed_selected),
        "selected_observed_delta_sum": observed_delta_sum,
        "selected_observed_delta_mean": observed_delta_sum / len(observed_selected) if observed_selected else 0.0,
        "estimated_delta_per_test_row": observed_delta_sum / len(rows) if rows else 0.0,
        "realized_loss_count": len(observed_loss_values),
        "realized_loss_mean": sum(observed_loss_values) / len(observed_loss_values) if observed_loss_values else 0.0,
        "realized_max_loss": max(observed_loss_values) if observed_loss_values else 0.0,
    }


def threshold_rows_for_run(label: str, path: Path, thresholds: Iterable[float]) -> list[dict[str, Any]]:
    manifest = read_manifest(path)
    rows = test_rows(read_csv(path / PREDICTIONS_FILE))
    rows_by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_source.setdefault(row_source(row, manifest), []).append(row)
    out: list[dict[str, Any]] = []
    for source, source_rows in sorted(rows_by_source.items()):
        for threshold in thresholds:
            out.append(
                threshold_metrics_for_source(
                    label=label,
                    path=path,
                    threshold=threshold,
                    source=source,
                    rows=source_rows,
                )
            )
    return out


def summarize_threshold_stability(
    rows: list[dict[str, Any]],
    *,
    min_selected_per_source: int,
    min_sources_with_selection: int,
    require_all_sources_nonnegative: bool,
) -> list[dict[str, Any]]:
    by_threshold: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        by_threshold.setdefault(safe_float(row.get("threshold")), []).append(row)
    summaries: list[dict[str, Any]] = []
    for threshold, threshold_rows in sorted(by_threshold.items()):
        eligible = [
            row for row in threshold_rows if safe_int(row.get("selected_rows")) >= min_selected_per_source
        ]
        selected_sources = len(eligible)
        negative_sources = [
            row for row in eligible if safe_float(row.get("estimated_delta_per_test_row")) < 0.0
        ]
        total_rows = sum(safe_int(row.get("rows")) for row in threshold_rows)
        total_selected = sum(safe_int(row.get("selected_rows")) for row in threshold_rows)
        total_observed = sum(safe_int(row.get("selected_observed_rows")) for row in threshold_rows)
        total_delta = sum(safe_float(row.get("selected_observed_delta_sum")) for row in threshold_rows)
        enough_sources = selected_sources >= min_sources_with_selection
        nonnegative_ok = not require_all_sources_nonnegative or not negative_sources
        enough_rows = all(safe_int(row.get("selected_rows")) >= min_selected_per_source for row in eligible)
        stable = enough_sources and nonnegative_ok and enough_rows
        estimated_values = [safe_float(row.get("estimated_delta_per_test_row")) for row in eligible]
        summaries.append(
            {
                "threshold": threshold,
                "source_count": len(threshold_rows),
                "sources_with_min_selected": selected_sources,
                "total_test_rows": total_rows,
                "total_selected_rows": total_selected,
                "total_observed_selected_rows": total_observed,
                "total_observed_delta_sum": total_delta,
                "aggregate_estimated_delta_per_test_row": total_delta / total_rows if total_rows else 0.0,
                "eligible_mean_estimated_delta_per_test_row": (
                    sum(estimated_values) / len(estimated_values) if estimated_values else 0.0
                ),
                "eligible_min_estimated_delta_per_test_row": min(estimated_values) if estimated_values else 0.0,
                "eligible_max_estimated_delta_per_test_row": max(estimated_values) if estimated_values else 0.0,
                "negative_source_count": len(negative_sources),
                "negative_sources": ";".join(str(row.get("source") or row.get("run") or "") for row in negative_sources),
                "min_selected_per_source": min_selected_per_source,
                "min_sources_with_selection": min_sources_with_selection,
                "stable_candidate": int(stable),
                "decision": "Candidate" if stable else "No-Go",
                "no_go_reason": "" if stable else no_go_reason(enough_sources, nonnegative_ok, selected_sources),
            }
        )
    return summaries


def no_go_reason(enough_sources: bool, nonnegative_ok: bool, selected_sources: int) -> str:
    if not enough_sources:
        return f"too_few_sources_with_selection:{selected_sources}"
    if not nonnegative_ok:
        return "negative_source_delta"
    return "unstable"


def analyze_runs(
    runs: list[tuple[str, Path]],
    *,
    thresholds: Iterable[float],
    min_selected_per_source: int,
    min_sources_with_selection: int,
    require_all_sources_nonnegative: bool = True,
) -> dict[str, Any]:
    thresholds = tuple(thresholds)
    source_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    for label, path in runs:
        source_rows.append(source_test_metrics(label, path))
        threshold_rows.extend(threshold_rows_for_run(label, path, thresholds))
    stability_rows = summarize_threshold_stability(
        threshold_rows,
        min_selected_per_source=min_selected_per_source,
        min_sources_with_selection=min_sources_with_selection,
        require_all_sources_nonnegative=require_all_sources_nonnegative,
    )
    candidate_rows = [row for row in stability_rows if safe_int(row.get("stable_candidate"))]
    decision = "Needs runtime validation" if candidate_rows else "No-Go"
    if candidate_rows:
        best = max(candidate_rows, key=lambda row: safe_float(row.get("eligible_min_estimated_delta_per_test_row")))
        reason = f"stable_threshold:{best['threshold']}"
    else:
        diagnostic_rows = [
            row
            for row in stability_rows
            if safe_int(row.get("sources_with_min_selected")) >= min_sources_with_selection
        ]
        if not diagnostic_rows:
            diagnostic_rows = stability_rows
        best = max(
            diagnostic_rows,
            key=lambda row: (
                safe_float(row.get("eligible_min_estimated_delta_per_test_row")),
                safe_float(row.get("aggregate_estimated_delta_per_test_row")),
            ),
            default={},
        )
        reason = "no_threshold_positive_across_sources"
    return {
        "source_rows": source_rows,
        "threshold_rows": threshold_rows,
        "stability_rows": stability_rows,
        "decision": decision,
        "reason": reason,
        "best_threshold": best,
    }


def write_markdown(path: Path, analysis: dict[str, Any]) -> None:
    source_rows = analysis["source_rows"]
    stability_rows = analysis["stability_rows"]
    decision = str(analysis.get("decision") or "No-Go")
    reason = str(analysis.get("reason") or "")
    lines = [
        "# HU T2 Stage8c Source-Heldout Analysis",
        "",
        "This artifact treats source-seed heldout performance as the validation unit. It does not approve runtime use, P2 fixed status, T1, production, or 50k teacher generation.",
        "",
        "## Decision",
        "",
        f"- selector promotion: `{decision}`",
        f"- reason: `{reason}`",
        "- production / P2 fixed / T1 / 50k: `No-Go`",
        "",
        "## Heldout Sources",
        "",
        "| run | source | rows | positives | observed | AP | AUC | Brier |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in source_rows:
        lines.append(
            "| {run} | {source} | {rows} | {pos} | {obs} | {ap:.4f} | {auc:.4f} | {brier:.4f} |".format(
                run=row["run"],
                source=row["test_sources"],
                rows=row["test_rows"],
                pos=row["test_positive_rows"],
                obs=row["test_observed_rows"],
                ap=safe_float(row["test_average_precision"]),
                auc=safe_float(row["test_roc_auc"]),
                brier=safe_float(row["test_brier"]),
            )
        )
    lines.extend(
        [
            "",
            "## Threshold Stability",
            "",
            "| threshold | sources with min selected | selected | aggregate EV/row | min source EV/row | negative sources | decision |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in stability_rows:
        lines.append(
            "| {threshold:.3f} | {sources} | {selected} | {agg:.4f} | {min_ev:.4f} | {neg} | {decision} |".format(
                threshold=safe_float(row["threshold"]),
                sources=safe_int(row["sources_with_min_selected"]),
                selected=safe_int(row["total_selected_rows"]),
                agg=safe_float(row["aggregate_estimated_delta_per_test_row"]),
                min_ev=safe_float(row["eligible_min_estimated_delta_per_test_row"]),
                neg=safe_int(row["negative_source_count"]),
                decision=row["decision"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A threshold is only a candidate if it selects enough rows on enough heldout sources and no eligible heldout source has negative estimated realized delta per test row.",
            "- If this report says `No-Go`, the model can still be used for replay target mining, but not as a runtime selector.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(output_dir: Path, analysis: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "source_heldout_run_metrics.csv", analysis["source_rows"])
    write_csv(output_dir / "source_heldout_threshold_by_source.csv", analysis["threshold_rows"])
    write_csv(output_dir / "source_heldout_threshold_stability.csv", analysis["stability_rows"])
    (output_dir / "source_heldout_decision.json").write_text(
        json.dumps(
            {
                "selector_promotion": analysis["decision"],
                "reason": analysis["reason"],
                "production_p2_fixed": "No-Go",
                "t1": "No-Go",
                "teacher_50k": "No-Go",
                "best_threshold": analysis.get("best_threshold") or {},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_markdown(output_dir / "source_heldout_summary.md", analysis)


def main() -> None:
    args = parse_args()
    runs = parse_run_specs(args.run)
    thresholds = tuple(args.threshold) if args.threshold else DEFAULT_THRESHOLDS
    analysis = analyze_runs(
        runs,
        thresholds=thresholds,
        min_selected_per_source=args.min_selected_per_source,
        min_sources_with_selection=args.min_sources_with_selection,
        require_all_sources_nonnegative=args.require_all_sources_nonnegative,
    )
    write_outputs(args.output_dir, analysis)
    print(json.dumps({"decision": analysis["decision"], "reason": analysis["reason"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
