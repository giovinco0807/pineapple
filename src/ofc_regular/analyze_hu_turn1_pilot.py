"""Analyze HU Turn1 pilot teacher data."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .hu_turn3_model import load_hu_action_value_model, sample_to_matrix
from .train_hu_turn1_candidate_generator import action_regrets, predict_scores


GAP_BUCKETS = (
    ("lt_0p01", 0.01),
    ("lt_0p05", 0.05),
    ("lt_0p10", 0.10),
    ("lt_0p25", 0.25),
    ("lt_0p50", 0.50),
    ("lt_1p00", 1.00),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--baseline-model", type=Path)
    parser.add_argument("--refinement-output", type=Path)
    parser.add_argument("--refinement-limit", type=int, default=0)
    parser.add_argument("--close-gap-threshold", type=float, default=0.50)
    parser.add_argument("--high-se-threshold", type=float, default=3.00)
    parser.add_argument("--high-regret-threshold", type=float, default=1.00)
    return parser.parse_args()


def read_records(path: Path) -> list[dict[str, Any]]:
    paths = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
    records: list[dict[str, Any]] = []
    for item in paths:
        if not item.exists():
            continue
        with item.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    return records


def finite_float(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def quantile(values: Iterable[float], q: float) -> float:
    data = np.asarray([float(value) for value in values], dtype=np.float64)
    if data.size == 0:
        return 0.0
    return float(np.quantile(data, q))


def mean(values: Iterable[float]) -> float:
    data = [float(value) for value in values]
    return float(sum(data) / len(data)) if data else 0.0


def action_key(action: dict[str, Any] | None) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    if not isinstance(action, dict):
        return ((), ())
    placements = tuple((str(card), str(row)) for card, row in action.get("placements", ()))
    discards = tuple(str(card) for card in action.get("discards", ()))
    return placements, discards


def best_score_gap(actions: list[dict[str, Any]]) -> tuple[float, float, float, int]:
    scores = sorted((as_float(action.get("score")) for action in actions), reverse=True)
    if not scores:
        return 0.0, 0.0, 0.0, 0
    best = scores[0]
    raw_gap = best - scores[1] if len(scores) > 1 else 0.0
    best_ties = sum(1 for score in scores if abs(score - best) <= 1e-9)
    second = next((score for score in scores if score < best - 1e-9), best)
    return best, raw_gap, best - second, best_ties


def sorted_action_indices(actions: list[dict[str, Any]]) -> list[int]:
    return sorted(
        range(len(actions)),
        key=lambda index: (-as_float(actions[index].get("score")), int(actions[index].get("original_index", index))),
    )


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "median": quantile(values, 0.5),
        "p90": quantile(values, 0.9),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts: list[float] = []
    score_gaps: list[float] = []
    computed_gaps: list[float] = []
    distinct_gaps: list[float] = []
    best_scores: list[float] = []
    action_eval_seconds: list[float] = []
    action_se_values: list[float] = []
    best_action_se_values: list[float] = []
    max_action_se_values: list[float] = []
    best_tie_counts: list[float] = []
    invalid_action_rows = 0
    invalid_state_rows = 0
    action_count_mismatches = 0
    best_action_missing = 0
    best_action_not_legal = 0
    best_score_mismatches = 0
    score_gap_mismatches = 0
    actions_truncated = 0
    schema_counts: Counter[str] = Counter()
    profile_counts: Counter[str] = Counter()
    opponent_profile_counts: Counter[str] = Counter()
    t2_profile_counts: Counter[str] = Counter()
    t3_counts: Counter[str] = Counter()
    seat_counts: Counter[str] = Counter()
    player_counts: Counter[str] = Counter()
    future_sample_counts: Counter[str] = Counter()
    sample_ids: Counter[str] = Counter()
    state_keys: Counter[str] = Counter()
    gap_bucket_counts: Counter[str] = Counter()

    for record in records:
        actions = record.get("actions")
        if not isinstance(actions, list) or not actions:
            invalid_state_rows += 1
            actions = []
        action_counts.append(float(len(actions)))
        declared_count = int(record.get("action_count", len(actions)) or 0)
        if declared_count != len(actions):
            action_count_mismatches += 1
        if bool(record.get("actions_truncated")):
            actions_truncated += 1

        schema_counts[str(record.get("schema", "unknown"))] += 1
        profile_counts[str(record.get("profile", "unknown"))] += 1
        opponent_profile_counts[str(record.get("opponent_profile", "unknown"))] += 1
        t2_profile_counts[str(record.get("t2_continuation_profile", "unknown"))] += 1
        t3_counts[str(record.get("t3_continuation", "unknown"))] += 1
        seat_counts[str(record.get("seat", "unknown"))] += 1
        player_counts[str(record.get("player", "unknown"))] += 1
        future_sample_counts[str(record.get("future_samples", "unknown"))] += 1
        sample_ids[str(record.get("sample_id", ""))] += 1
        state_keys[f"{record.get('hand_seed', '')}:{record.get('player', '')}"] += 1

        best_score, computed_gap, distinct_gap, best_ties = best_score_gap(actions)
        action_order = sorted_action_indices(actions)
        best_scores.append(best_score)
        computed_gaps.append(computed_gap)
        distinct_gaps.append(distinct_gap)
        best_tie_counts.append(float(best_ties))
        if action_order:
            best_action_se_values.append(as_float(actions[action_order[0]].get("se")))
            max_action_se_values.append(max(as_float(action.get("se")) for action in actions))
        declared_gap = as_float(record.get("score_gap"), computed_gap)
        score_gaps.append(declared_gap)
        if abs(declared_gap - computed_gap) > 1e-6:
            score_gap_mismatches += 1
        for name, threshold in GAP_BUCKETS:
            if declared_gap < threshold:
                gap_bucket_counts[name] += 1
        if declared_gap >= 1.0:
            gap_bucket_counts["ge_1p00"] += 1

        best_action = record.get("best_action")
        if best_action is None:
            best_action_missing += 1
        elif isinstance(best_action, int):
            if best_action < 0 or best_action >= len(actions):
                best_action_not_legal += 1
            elif abs(as_float(actions[best_action].get("score")) - best_score) > 1e-6:
                best_score_mismatches += 1
        elif isinstance(best_action, dict):
            best_key = action_key(best_action)
            legal_by_key = {action_key(action): action for action in actions}
            legal_best = legal_by_key.get(best_key)
            if legal_best is None:
                best_action_not_legal += 1
            elif abs(as_float(legal_best.get("score")) - best_score) > 1e-6:
                best_score_mismatches += 1
        else:
            best_action_missing += 1

        for action in actions:
            if not finite_float(action.get("score")) or not finite_float(action.get("ev")) or not finite_float(action.get("se")):
                invalid_action_rows += 1
            action_se_values.append(as_float(action.get("se")))
            action_eval_seconds.append(as_float(action.get("action_eval_seconds")))

    duplicate_sample_ids = sum(count - 1 for value, count in sample_ids.items() if value and count > 1)
    duplicate_state_keys = sum(count - 1 for value, count in state_keys.items() if value and count > 1)
    total_actions = int(sum(action_counts))
    first_count = int(seat_counts.get("first", 0))
    second_count = int(seat_counts.get("second", 0))
    seat_max = max(first_count, second_count)
    expected_seats_present = [seat for seat in ("first", "second") if seat_counts.get(seat, 0) > 0]
    missing_expected_seats = [seat for seat in ("first", "second") if seat_counts.get(seat, 0) <= 0]
    seat_balance_ratio = (min(first_count, second_count) / seat_max) if seat_max > 0 else 0.0
    seat_distribution_warning = ""
    if len(expected_seats_present) == 1:
        seat_distribution_warning = (
            "single_seat_only: T1 pilot rows include only one of first/second; "
            "use ShardSamples=2 for paired rows or a RecordSkipBase=1 second-only complement."
        )
    elif expected_seats_present and seat_balance_ratio < 0.80:
        seat_distribution_warning = "imbalanced_seats: first/second split is materially uneven."
    return {
        "schema": "hu_turn1_stage1_pilot_analysis_v1",
        "records": len(records),
        "total_actions": total_actions,
        "invalid_state_rows": invalid_state_rows,
        "invalid_action_rows": invalid_action_rows,
        "action_count_mismatches": action_count_mismatches,
        "best_action_missing": best_action_missing,
        "best_action_not_legal": best_action_not_legal,
        "best_score_mismatches": best_score_mismatches,
        "score_gap_mismatches": score_gap_mismatches,
        "actions_truncated": actions_truncated,
        "duplicate_sample_ids": duplicate_sample_ids,
        "duplicate_state_keys": duplicate_state_keys,
        "action_count": stats(action_counts),
        "score_gap": stats(score_gaps),
        "computed_score_gap": stats(computed_gaps),
        "distinct_score_gap": stats(distinct_gaps),
        "best_score": stats(best_scores),
        "best_tie_count": stats(best_tie_counts),
        "action_se": stats(action_se_values),
        "best_action_se": stats(best_action_se_values),
        "max_action_se": stats(max_action_se_values),
        "action_eval_seconds": stats(action_eval_seconds),
        "score_gap_buckets": dict(sorted(gap_bucket_counts.items())),
        "schema_counts": dict(sorted(schema_counts.items())),
        "profile_counts": dict(sorted(profile_counts.items())),
        "opponent_profile_counts": dict(sorted(opponent_profile_counts.items())),
        "t2_continuation_profile_counts": dict(sorted(t2_profile_counts.items())),
        "t3_continuation_counts": dict(sorted(t3_counts.items())),
        "seat_counts": dict(sorted(seat_counts.items())),
        "seat_distribution": {
            "first": first_count,
            "second": second_count,
            "missing_expected_seats": missing_expected_seats,
            "single_seat_only": len(expected_seats_present) == 1,
            "balance_ratio": seat_balance_ratio,
            "warning": seat_distribution_warning,
        },
        "player_counts": dict(sorted(player_counts.items())),
        "future_sample_counts": dict(sorted(future_sample_counts.items())),
    }


def _serializable_action_stub(action: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    return {
        "placements": action.get("placements", []),
        "discards": action.get("discards", []),
        "score": as_float(action.get("score")),
        "se": as_float(action.get("se")),
        "original_index": int(action.get("original_index", action.get("action_index", -1))),
        "action_index": int(action.get("action_index", action.get("original_index", -1))),
    }


def refinement_candidate_rows(
    records: list[dict[str, Any]],
    *,
    close_gap_threshold: float,
    high_se_threshold: float,
    high_regret_threshold: float,
    baseline_model_path: Path | None = None,
    limit: int = 0,
) -> list[dict[str, Any]]:
    baseline_model = load_hu_action_value_model(baseline_model_path) if baseline_model_path else None
    rows: list[dict[str, Any]] = []
    for row_index, record in enumerate(records):
        actions = record.get("actions")
        if not isinstance(actions, list) or not actions:
            continue
        action_order = sorted_action_indices(actions)
        if not action_order:
            continue
        best_index = int(action_order[0])
        best_action = actions[best_index]
        scores = np.asarray([as_float(action.get("score")) for action in actions], dtype=np.float64)
        regrets = float(np.max(scores)) - scores
        raw_gap = as_float(record.get("score_gap"))
        distinct_gap = best_score_gap(actions)[2]
        best_se = as_float(best_action.get("se"))
        max_se = max(as_float(action.get("se")) for action in actions)

        baseline_index: int | None = None
        baseline_regret = 0.0
        baseline_score = 0.0
        if baseline_model is not None:
            features, _targets = sample_to_matrix(record)
            predicted = predict_scores(baseline_model, features)
            baseline_index = int(np.argsort(-predicted, kind="mergesort")[0])
            baseline_regret = float(regrets[baseline_index])
            baseline_score = float(predicted[baseline_index])

        reasons: list[str] = []
        if distinct_gap <= close_gap_threshold:
            reasons.append("close_gap")
        if best_se >= high_se_threshold or max_se >= high_se_threshold:
            reasons.append("high_se")
        if baseline_model is not None and baseline_regret >= high_regret_threshold:
            reasons.append("high_baseline_regret")
        if not reasons:
            continue

        candidate = {
            "schema": "hu_turn1_stage1_refinement_candidate_v1",
            "row_index": row_index,
            "sample_id": record.get("sample_id"),
            "hand_seed": record.get("hand_seed"),
            "player": record.get("player"),
            "seat": record.get("seat"),
            "profile": record.get("profile"),
            "opponent_profile": record.get("opponent_profile"),
            "future_samples": record.get("future_samples"),
            "action_count": len(actions),
            "reasons": reasons,
            "priority_score": float(
                (baseline_regret * 4.0)
                + max(0.0, close_gap_threshold - distinct_gap)
                + max(0.0, max_se - high_se_threshold)
            ),
            "score_gap": raw_gap,
            "distinct_score_gap": distinct_gap,
            "best_score": as_float(best_action.get("score")),
            "best_action_index": best_index,
            "best_original_index": int(best_action.get("original_index", best_action.get("action_index", best_index))),
            "best_action_se": best_se,
            "max_action_se": max_se,
            "baseline_model": str(baseline_model_path) if baseline_model_path else "",
            "baseline_top1_action_index": baseline_index,
            "baseline_top1_regret": baseline_regret,
            "baseline_top1_score": baseline_score,
            "board": record.get("board"),
            "opponent_board": record.get("opponent_board"),
            "dealt": record.get("dealt"),
            "visible_dead_cards": record.get("visible_dead_cards", record.get("dead_cards", [])),
            "hero_private_discards": record.get("hero_private_discards", []),
            "opponent_private_discards": record.get("opponent_private_discards", []),
            "best_action": _serializable_action_stub(best_action),
            "baseline_action": _serializable_action_stub(actions[baseline_index]) if baseline_index is not None else {},
            "top_action_indices": [int(index) for index in action_order[: min(5, len(action_order))]],
        }
        rows.append(candidate)
    rows.sort(key=lambda row: (-float(row["priority_score"]), str(row.get("hand_seed")), int(row.get("player") or 0)))
    if limit > 0:
        rows = rows[:limit]
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def metric_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, value in summary.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, dict):
                    for stat_name, stat_value in subvalue.items():
                        rows.append({"group": key, "metric": f"{subkey}_{stat_name}", "value": stat_value})
                else:
                    rows.append({"group": key, "metric": str(subkey), "value": subvalue})
        elif key != "schema":
            rows.append({"group": "summary", "metric": key, "value": value})
    return rows


def markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# HU Turn1 Stage1 Pilot Analysis",
        "",
        "## Completeness",
        "",
        f"- records: `{summary['records']}`",
        f"- total actions: `{summary['total_actions']}`",
        f"- invalid state rows: `{summary['invalid_state_rows']}`",
        f"- invalid action rows: `{summary['invalid_action_rows']}`",
        f"- action count mismatches: `{summary['action_count_mismatches']}`",
        f"- best action not legal: `{summary['best_action_not_legal']}`",
        f"- score gap mismatches: `{summary['score_gap_mismatches']}`",
        f"- actions truncated: `{summary['actions_truncated']}`",
        f"- duplicate local sample ids: `{summary['duplicate_sample_ids']}`",
        f"- duplicate state keys: `{summary['duplicate_state_keys']}`",
        "",
        "## Core Stats",
        "",
        f"- action count mean/median/p95/max: `{summary['action_count']['mean']:.3f}` / `{summary['action_count']['median']:.3f}` / `{summary['action_count']['p95']:.3f}` / `{summary['action_count']['max']:.3f}`",
        f"- score gap mean/median/p25? see JSON; p90/p95/max: `{summary['score_gap']['mean']:.4f}` / `{summary['score_gap']['median']:.4f}` / `{summary['score_gap']['p90']:.4f}` / `{summary['score_gap']['p95']:.4f}` / `{summary['score_gap']['max']:.4f}`",
        f"- distinct score gap mean/median/p95/max: `{summary['distinct_score_gap']['mean']:.4f}` / `{summary['distinct_score_gap']['median']:.4f}` / `{summary['distinct_score_gap']['p95']:.4f}` / `{summary['distinct_score_gap']['max']:.4f}`",
        f"- best tie count mean/max: `{summary['best_tie_count']['mean']:.3f}` / `{summary['best_tie_count']['max']:.3f}`",
        f"- action SE mean/p90/max: `{summary['action_se']['mean']:.4f}` / `{summary['action_se']['p90']:.4f}` / `{summary['action_se']['max']:.4f}`",
        f"- best-action SE mean/p90/max: `{summary['best_action_se']['mean']:.4f}` / `{summary['best_action_se']['p90']:.4f}` / `{summary['best_action_se']['max']:.4f}`",
        "",
        "## Profiles",
        "",
        f"- profile: `{summary['profile_counts']}`",
        f"- opponent profile: `{summary['opponent_profile_counts']}`",
        f"- T2 continuation profile: `{summary['t2_continuation_profile_counts']}`",
        f"- T3 continuation: `{summary['t3_continuation_counts']}`",
        f"- seat counts: `{summary['seat_counts']}`",
        f"- seat balance ratio: `{summary['seat_distribution']['balance_ratio']:.3f}`",
        f"- seat warning: `{summary['seat_distribution']['warning'] or 'none'}`",
        "",
        "## Status",
        "",
        "- production / P2 fixed: `No-Go`",
        "- T1 2k-5k pilot teacher: `Go` if the completeness counters above stay clean",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    records = read_records(args.input)
    summary = summarize(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_output = args.summary_output or args.output_dir / "summary.json"
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "summary.md").write_text(markdown_summary(summary), encoding="utf-8")
    write_csv(args.output_dir / "metrics.csv", metric_rows(summary))
    if args.refinement_output:
        refinement_rows = refinement_candidate_rows(
            records,
            close_gap_threshold=args.close_gap_threshold,
            high_se_threshold=args.high_se_threshold,
            high_regret_threshold=args.high_regret_threshold,
            baseline_model_path=args.baseline_model,
            limit=args.refinement_limit,
        )
        write_jsonl(args.refinement_output, refinement_rows)
        write_csv(args.output_dir / "refinement_candidates.csv", refinement_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
