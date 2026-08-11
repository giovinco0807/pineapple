"""Compare Stage8c prediction rankers on labeled TopK-confirm rows.

This diagnostic joins fire-head probabilities and delta-head predictions for
the same state/action rows, then evaluates topK ranking by realized delta. It
is for replay triage only, not runtime approval.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fire-predictions", type=Path, required=True)
    parser.add_argument("--delta-prediction", action="append", default=[], help="name=delta_head_predictions.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-recommended-use", action="append", default=["topk_confirm_topk_empty"])
    parser.add_argument("--topk", default="1,3,5,10,20,50,100")
    parser.add_argument("--top-row-limit", type=int, default=100, help="Top rows to write per ranker for the test split.")
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def read_csv(path: Path) -> list[dict[str, str]]:
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


def row_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("source_log", "")),
        str(row.get("hand_seed", "")),
        str(row.get("candidate_index", "")),
        str(row.get("baseline_index", "")),
        str(row.get("recommended_training_use", "")),
    )


def parse_named_path(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"expected name=path: {spec}")
    name, raw_path = spec.split("=", 1)
    name = name.strip()
    path = Path(raw_path.strip())
    if not name:
        raise SystemExit(f"empty prediction name: {spec}")
    if not path.exists():
        raise SystemExit(f"prediction path does not exist: {path}")
    return name, path


def indexed_rows(path: Path) -> dict[tuple[str, str, str, str, str], dict[str, str]]:
    rows = read_csv(path)
    result: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    duplicates: list[tuple[str, str, str, str, str]] = []
    for row in rows:
        key = row_key(row)
        if key in result:
            duplicates.append(key)
        result[key] = row
    if duplicates:
        raise SystemExit(f"{path} has duplicate join keys; first duplicate={duplicates[0]}")
    return result


def zscore(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    std = float(np.std(values))
    if std < 1e-9:
        return np.zeros_like(values, dtype=np.float64)
    return (values - float(np.mean(values))) / std


def joined_rows(
    fire_path: Path,
    delta_specs: list[str],
    *,
    exclude_recommended_use: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fire_rows = [row for row in read_csv(fire_path) if row.get("recommended_training_use", "") not in exclude_recommended_use]
    delta_maps: dict[str, dict[tuple[str, str, str, str, str], dict[str, str]]] = {}
    delta_paths: dict[str, str] = {}
    for spec in delta_specs:
        name, path = parse_named_path(spec)
        delta_maps[name] = indexed_rows(path)
        delta_paths[name] = str(path)

    rows: list[dict[str, Any]] = []
    join_counts = {name: 0 for name in delta_maps}
    for fire in fire_rows:
        key = row_key(fire)
        matched_deltas: dict[str, dict[str, str] | None] = {
            name: mapping.get(key) for name, mapping in delta_maps.items()
        }
        split_name = fire.get("split") or ""
        if not split_name:
            for delta in matched_deltas.values():
                if delta and delta.get("split"):
                    split_name = delta["split"]
                    break
        row: dict[str, Any] = {
            "join_key": "|".join(key),
            "split": split_name,
            "source_log": fire.get("source_log", ""),
            "config_id": fire.get("config_id", ""),
            "hand_seed": fire.get("hand_seed", ""),
            "seat": fire.get("seat", ""),
            "seat_swap": fire.get("seat_swap", ""),
            "candidate_source": fire.get("candidate_source", ""),
            "recommended_training_use": fire.get("recommended_training_use", ""),
            "risk_target_group": fire.get("risk_target_group", fire.get("recommended_training_use", "")),
            "state_signature": fire.get("state_signature", ""),
            "action_signature": fire.get("action_signature", ""),
            "baseline_action_signature": fire.get("baseline_action_signature", ""),
            "candidate_index": fire.get("candidate_index", ""),
            "baseline_index": fire.get("baseline_index", ""),
            "realized_delta_observed": fire.get("realized_delta_observed", ""),
            "local_replay_status": fire.get("local_replay_status", ""),
            "local_replay_label": fire.get("local_replay_label", ""),
            "realized_delta": safe_float(fire.get("realized_delta")),
            "realized_loss": safe_float(fire.get("realized_loss"), max(0.0, -safe_float(fire.get("realized_delta")))),
            "risk_probability": safe_float(fire.get("risk_probability")),
            "predicted_delta": safe_float(fire.get("predicted_delta")),
            "confirm_delta": safe_float(fire.get("confirm_delta")),
            "confirm_delta_se": safe_float(fire.get("confirm_delta_se")),
            "candidate_ev_rank": safe_float(fire.get("candidate_ev_rank"), 9999.0),
        }
        for name, delta in matched_deltas.items():
            if delta is not None:
                join_counts[name] += 1
                row[f"{name}_delta_prediction"] = safe_float(delta.get("delta_prediction"))
                row[f"{name}_split"] = delta.get("split", "")
            else:
                row[f"{name}_delta_prediction"] = None
                row[f"{name}_split"] = ""
        rows.append(row)
    manifest = {
        "fire_predictions": str(fire_path),
        "fire_rows_after_filter": len(fire_rows),
        "delta_predictions": delta_paths,
        "joined_delta_counts": join_counts,
        "exclude_recommended_use": sorted(exclude_recommended_use),
    }
    return rows, manifest


def score_columns(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    columns: list[tuple[str, str]] = [
        ("fire_probability", "risk_probability"),
        ("predicted_delta", "predicted_delta"),
        ("confirm_delta", "confirm_delta"),
    ]
    for key in sorted(rows[0] if rows else {}):
        if key.endswith("_delta_prediction"):
            columns.append((key.removesuffix("_delta_prediction"), key))
    return columns


def ranker_role(ranker_name: str) -> dict[str, Any]:
    """Classify whether a ranker can be used at runtime.

    The comparison intentionally mixes deployable model scores with replay
    diagnostics.  Keep that distinction explicit in machine-readable outputs so
    confirm/replay-derived scores cannot be accidentally promoted as runtime
    gates.
    """

    parts = [part.strip().lower() for part in ranker_name.split("+") if part.strip()]
    joined = "+".join(parts)
    if any("confirm_delta" == part or part.startswith("confirm") for part in parts):
        return {
            "ranker_role": "replay_triage_only",
            "runtime_eligible": False,
            "role_reason": "uses confirm/replay delta; valid for triage ordering, not runtime deployment evidence",
        }
    if any(token in joined for token in ("teacher", "oracle", "observed", "realized", "replay", "high_mc")):
        return {
            "ranker_role": "diagnostic_only",
            "runtime_eligible": False,
            "role_reason": "uses observed or replay-derived information unavailable in normal runtime",
        }
    if "+" in ranker_name:
        return {
            "ranker_role": "diagnostic_combo",
            "runtime_eligible": False,
            "role_reason": "normalized score combination is analysis-only until implemented and validated as a runtime feature",
        }
    return {
        "ranker_role": "deployable_ranker_input",
        "runtime_eligible": True,
        "role_reason": "uses model/runtime prediction columns only; still requires realized validation before promotion",
    }


def recommendation_for_ranker(row: dict[str, Any]) -> str:
    role = str(row.get("ranker_role", ""))
    if role == "deployable_ranker_input":
        if safe_float(row.get("selected_delta_mean")) > 0.0 and safe_float(row.get("selected_positive_rate")) >= 0.5:
            return "candidate_generator_replay_triage_go"
        return "candidate_generator_needs_more_evidence"
    if role == "replay_triage_only":
        return "replay_target_selection_only"
    if role == "diagnostic_combo":
        return "analysis_only_combo_not_runtime"
    return "diagnostic_only_not_runtime"


def evaluate_rankers(
    rows: list[dict[str, Any]],
    topk_values: list[int],
    *,
    top_row_limit: int = 100,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    scored_rows: list[dict[str, Any]] = []
    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_split.setdefault(str(row.get("split", "")), []).append(row)

    base_columns = score_columns(rows)
    for split_name, split_rows in sorted(by_split.items()):
        if not split_rows:
            continue
        raw_score_arrays: dict[str, np.ndarray] = {}
        valid_masks: dict[str, np.ndarray] = {}
        for ranker_name, column in base_columns:
            values = np.asarray([
                safe_float(row.get(column)) if row.get(column) is not None else np.nan for row in split_rows
            ], dtype=np.float64)
            valid = np.isfinite(values)
            if np.any(valid):
                raw_score_arrays[ranker_name] = values
                valid_masks[ranker_name] = valid
        # Pairwise normalized combinations. These are diagnostic only.
        combo_scores: dict[str, np.ndarray] = {}
        for left in raw_score_arrays:
            for right in raw_score_arrays:
                if left >= right:
                    continue
                valid = valid_masks[left] & valid_masks[right]
                if not np.any(valid):
                    continue
                combo = np.full(len(split_rows), np.nan, dtype=np.float64)
                combo[valid] = zscore(raw_score_arrays[left][valid]) + zscore(raw_score_arrays[right][valid])
                combo_scores[f"{left}+{right}"] = combo
        all_scores = raw_score_arrays | combo_scores
        for ranker_name, values in sorted(all_scores.items()):
            valid = np.isfinite(values)
            if not np.any(valid):
                continue
            role = ranker_role(ranker_name)
            indices = np.where(valid)[0]
            order = indices[np.argsort(-values[valid])]
            targets = np.asarray([safe_float(row["realized_delta"]) for row in split_rows], dtype=np.float64)
            for topk in topk_values:
                k = min(int(topk), order.size)
                if k <= 0:
                    continue
                selected_idx = order[:k]
                selected = targets[selected_idx]
                losses = np.maximum(0.0, -selected)
                metric_rows.append(
                    {
                        "split": split_name,
                        "ranker": ranker_name,
                        "eligible_rows": int(order.size),
                        "topk": int(k),
                        "selected_delta_sum": float(np.sum(selected)),
                        "selected_delta_mean": float(np.mean(selected)),
                        "selected_positive_rate": float(np.mean(selected > 0.0)),
                        "selected_negative_count": int(np.sum(selected < 0.0)),
                        "selected_max_loss": float(np.max(losses)) if losses.size else 0.0,
                        "min_score": float(np.min(values[selected_idx])),
                        **role,
                    }
                )
            for rank, index in enumerate(order[: min(max(0, top_row_limit), order.size)], start=1):
                if split_name == "test":
                    scored_rows.append(split_rows[index] | {"ranker": ranker_name, "rank": rank, "ranker_score": float(values[index]), **role})
    return metric_rows, scored_rows


def recommendation_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Choose a compact, auditable row per split/ranker.

    This is a recommendation for what the ranker may be used for next, not a
    promotion decision.
    """

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in metric_rows:
        grouped.setdefault((str(row.get("split", "")), str(row.get("ranker", ""))), []).append(row)
    rows: list[dict[str, Any]] = []
    for (split_name, ranker_name), candidates in sorted(grouped.items()):
        candidates.sort(
            key=lambda row: (
                safe_float(row.get("selected_delta_mean")),
                safe_float(row.get("selected_positive_rate")),
                -safe_float(row.get("selected_max_loss")),
            ),
            reverse=True,
        )
        best = candidates[0]
        rows.append(
            {
                "split": split_name,
                "ranker": ranker_name,
                "ranker_role": best.get("ranker_role", ""),
                "runtime_eligible": best.get("runtime_eligible", False),
                "role_reason": best.get("role_reason", ""),
                "best_topk": best.get("topk", ""),
                "eligible_rows": best.get("eligible_rows", ""),
                "selected_delta_mean": best.get("selected_delta_mean", 0.0),
                "selected_delta_sum": best.get("selected_delta_sum", 0.0),
                "selected_positive_rate": best.get("selected_positive_rate", 0.0),
                "selected_negative_count": best.get("selected_negative_count", 0),
                "selected_max_loss": best.get("selected_max_loss", 0.0),
                "recommendation": recommendation_for_ranker(best),
            }
        )
    return rows


def write_summary(path: Path, metric_rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    recommendations = [row for row in recommendation_rows(metric_rows) if row.get("split") == "test"]
    test_rows = [row for row in metric_rows if row.get("split") == "test" and int(row.get("topk", 0)) in {3, 5, 10, 20}]
    test_rows.sort(key=lambda row: (int(row["topk"]), -safe_float(row["selected_delta_mean"])))
    lines = [
        "# HU T2 Stage8c Prediction Ranker Comparison",
        "",
        "This is replay-triage analysis only. It does not approve runtime use, P2 fixed status, T1, or 50k teacher generation.",
        "",
        "## Join",
        "",
        f"- fire rows after filter: `{manifest['fire_rows_after_filter']}`",
        f"- joined delta counts: `{json.dumps(manifest['joined_delta_counts'], sort_keys=True)}`",
        "",
        "## Test TopK",
        "",
        "| topK | ranker | role | runtime eligible | eligible | mean delta | sum delta | positive rate | max loss |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in test_rows:
        lines.append(
            "| {topk} | {ranker} | {ranker_role} | {runtime_eligible} | {eligible_rows} | {selected_delta_mean:+.4f} | {selected_delta_sum:+.4f} | {selected_positive_rate:.3f} | {selected_max_loss:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Recommendations",
            "",
            "| ranker | role | runtime eligible | best topK | mean delta | positive rate | recommendation |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in recommendations:
        lines.append(
            "| {ranker} | {ranker_role} | {runtime_eligible} | {best_topk} | {selected_delta_mean:+.4f} | {selected_positive_rate:.3f} | `{recommendation}` |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- execution: `Pass`",
            "- allowed role: `ranking/triage for selecting rows for expensive replay`",
            "- runtime gate: `No-Go`",
            "- production / P2 fixed: `No-Go`",
            "- 50k teacher: `No-Go`",
            "- T1 training: `No-Go`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    topk_values = [int(part.strip()) for part in args.topk.split(",") if part.strip()]
    rows, manifest = joined_rows(
        args.fire_predictions,
        args.delta_prediction,
        exclude_recommended_use=set(args.exclude_recommended_use or []),
    )
    metric_rows, top_rows = evaluate_rankers(rows, topk_values, top_row_limit=args.top_row_limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "prediction_ranker_metrics.csv", metric_rows)
    write_csv(args.output_dir / "prediction_ranker_top_rows.csv", top_rows)
    write_csv(args.output_dir / "prediction_ranker_recommendations.csv", recommendation_rows(metric_rows))
    (args.output_dir / "prediction_ranker_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary(args.output_dir / "prediction_ranker_comparison.md", metric_rows, manifest)
    print(json.dumps({"output_dir": str(args.output_dir), "metric_rows": len(metric_rows), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
