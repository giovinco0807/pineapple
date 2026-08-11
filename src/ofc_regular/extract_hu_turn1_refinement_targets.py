"""Select HU Turn1 pilot states for stronger refinement relabeling."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .hu_turn3_model import load_hu_action_value_model, sample_to_matrix as hu_sample_to_matrix
from .turn3_model import load_action_value_model, sample_to_matrix as self_sample_to_matrix


@dataclass(frozen=True)
class ScoredRecord:
    index: int
    record: dict[str, Any]
    state_key: str
    seat: str
    score_gap: float
    distinct_score_gap: float
    best_score: float
    best_tie_count: int
    best_action_se: float
    max_action_se: float
    max_model_regret: float
    model_metrics: dict[str, Any]
    selection_reasons: tuple[str, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--self-model", type=Path)
    parser.add_argument("--hu-model", type=Path)
    parser.add_argument("--high-regret-count", type=int, default=80)
    parser.add_argument("--high-se-count", type=int, default=0)
    parser.add_argument("--high-se-threshold", type=float, default=3.0)
    parser.add_argument("--high-gap-count", type=int, default=60)
    parser.add_argument("--low-gap-count", type=int, default=40)
    parser.add_argument("--random-count", type=int, default=20)
    parser.add_argument("--max-targets", type=int, default=200)
    parser.add_argument("--low-gap-threshold", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026062601)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def state_key(record: dict[str, Any]) -> str:
    payload = {
        "hand_seed": record.get("hand_seed"),
        "player": record.get("player"),
        "seat": record.get("seat"),
        "board": record.get("board"),
        "opponent_board": record.get("opponent_board"),
        "dealt": record.get("dealt"),
        "dead_cards": record.get("dead_cards"),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def scores(record: dict[str, Any]) -> list[float]:
    return [float(action.get("score", action.get("ev", 0.0)) or 0.0) for action in record.get("actions", ())]


def score_stats(record: dict[str, Any]) -> tuple[float, float, float, int]:
    values = sorted(scores(record), reverse=True)
    if not values:
        return 0.0, 0.0, 0.0, 0
    best = values[0]
    raw_gap = best - values[1] if len(values) > 1 else 0.0
    second_distinct = next((value for value in values if value < best - 1e-9), best)
    ties = sum(1 for value in values if abs(value - best) <= 1e-9)
    return best, raw_gap, best - second_distinct, ties


def se_stats(record: dict[str, Any]) -> tuple[float, float]:
    actions = record.get("actions", ()) or ()
    if not actions:
        return 0.0, 0.0
    values = scores(record)
    best_index = int(np.argmax(np.asarray(values, dtype=np.float64))) if values else 0
    se_values = [float(action.get("se", 0.0) or 0.0) for action in actions]
    return float(se_values[best_index]), float(max(se_values))


def model_regret(
    record: dict[str, Any],
    *,
    model: Any,
    matrix_fn: Callable[[dict[str, Any]], tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    features, targets = matrix_fn(record)
    predictions = model.predict_matrix(features)
    predicted_index = int(np.argmax(predictions))
    best_score = float(np.max(targets))
    predicted_score = float(targets[predicted_index])
    top_k = min(3, len(targets))
    top3 = set(np.argpartition(predictions, -top_k)[-top_k:].tolist())
    return {
        "predicted_index": predicted_index,
        "predicted_score": predicted_score,
        "predicted_model_score": float(predictions[predicted_index]),
        "regret": best_score - predicted_score,
        "top3_hit": bool(any(float(targets[index]) >= best_score - 1e-9 for index in top3)),
    }


def score_records(
    records: list[dict[str, Any]],
    *,
    self_model_path: Path | None,
    hu_model_path: Path | None,
) -> list[ScoredRecord]:
    model_specs: list[tuple[str, Any, Callable[[dict[str, Any]], tuple[np.ndarray, np.ndarray]]]] = []
    if self_model_path is not None:
        model_specs.append(("self_model", load_action_value_model(self_model_path), self_sample_to_matrix))
    if hu_model_path is not None:
        model_specs.append(("hu_model", load_hu_action_value_model(hu_model_path), hu_sample_to_matrix))

    scored: list[ScoredRecord] = []
    for index, record in enumerate(records):
        best, raw_gap, distinct_gap, ties = score_stats(record)
        best_se, max_se = se_stats(record)
        metrics: dict[str, Any] = {}
        regrets: list[float] = []
        for name, model, matrix_fn in model_specs:
            result = model_regret(record, model=model, matrix_fn=matrix_fn)
            metrics[name] = result
            regrets.append(float(result["regret"]))
        scored.append(
            ScoredRecord(
                index=index,
                record=record,
                state_key=state_key(record),
                seat=str(record.get("seat", "unknown")),
                score_gap=raw_gap,
                distinct_score_gap=distinct_gap,
                best_score=best,
                best_tie_count=ties,
                best_action_se=best_se,
                max_action_se=max_se,
                max_model_regret=max(regrets) if regrets else 0.0,
                model_metrics=metrics,
            )
        )
    return scored


def take_balanced(
    candidates: list[ScoredRecord],
    *,
    count: int,
    reason: str,
    selected: dict[str, ScoredRecord],
) -> None:
    if count <= 0:
        return
    per_seat = max(1, count // 2)
    seats = ("first", "second")
    taken = 0
    for seat in seats:
        seat_taken = 0
        for item in candidates:
            if item.seat != seat:
                continue
            selected[item.state_key] = (
                merge_reason(selected[item.state_key], reason)
                if item.state_key in selected
                else append_reason(item, reason)
            )
            seat_taken += 1
            taken += 1
            if seat_taken >= per_seat:
                break
    if taken < count:
        for item in candidates:
            selected[item.state_key] = (
                merge_reason(selected[item.state_key], reason)
                if item.state_key in selected
                else append_reason(item, reason)
            )
            taken += 1
            if taken >= count:
                break


def append_reason(item: ScoredRecord, reason: str) -> ScoredRecord:
    return ScoredRecord(
        index=item.index,
        record=item.record,
        state_key=item.state_key,
        seat=item.seat,
        score_gap=item.score_gap,
        distinct_score_gap=item.distinct_score_gap,
        best_score=item.best_score,
        best_tie_count=item.best_tie_count,
        best_action_se=item.best_action_se,
        max_action_se=item.max_action_se,
        max_model_regret=item.max_model_regret,
        model_metrics=item.model_metrics,
        selection_reasons=tuple(dict.fromkeys((*item.selection_reasons, reason))),
    )


def merge_reason(existing: ScoredRecord, reason: str) -> ScoredRecord:
    return append_reason(existing, reason)


def select_targets(
    scored: list[ScoredRecord],
    *,
    high_regret_count: int,
    high_se_count: int,
    high_se_threshold: float,
    high_gap_count: int,
    low_gap_count: int,
    random_count: int,
    low_gap_threshold: float,
    max_targets: int,
    seed: int,
) -> list[ScoredRecord]:
    selected: dict[str, ScoredRecord] = {}
    high_regret = sorted(scored, key=lambda item: item.max_model_regret, reverse=True)
    high_se = sorted(
        [item for item in scored if item.max_action_se >= high_se_threshold],
        key=lambda item: item.max_action_se,
        reverse=True,
    )
    high_gap = sorted(scored, key=lambda item: item.score_gap, reverse=True)
    low_gap = [item for item in scored if item.score_gap < low_gap_threshold]
    rng = random.Random(seed)
    random_pool = list(scored)
    rng.shuffle(low_gap)
    rng.shuffle(random_pool)

    take_balanced(
        high_regret,
        count=high_regret_count,
        reason="high_model_regret",
        selected=selected,
    )
    take_balanced(high_se, count=high_se_count, reason="high_action_se", selected=selected)
    take_balanced(high_gap, count=high_gap_count, reason="high_score_gap", selected=selected)
    take_balanced(low_gap, count=low_gap_count, reason="low_score_gap", selected=selected)
    take_balanced(random_pool, count=random_count, reason="random_cover", selected=selected)
    if len(selected) < max_targets:
        for item in random_pool:
            if item.state_key in selected:
                continue
            selected[item.state_key] = append_reason(item, "fill_cover")
            if len(selected) >= max_targets:
                break

    ranked = sorted(
        selected.values(),
        key=lambda item: (
            "high_model_regret" not in item.selection_reasons,
            -item.max_model_regret,
            -item.score_gap,
            item.index,
        ),
    )
    return ranked[:max_targets] if max_targets > 0 else ranked


def output_record(item: ScoredRecord, target_id: int) -> dict[str, Any]:
    record = dict(item.record)
    record.update(
        {
            "schema": "hu_turn1_stage1_refinement_target_v1",
            "target_id": target_id,
            "source_index": item.index,
            "source_schema": item.record.get("schema"),
            "state_key": item.state_key,
            "selection_reasons": list(item.selection_reasons),
            "selection_score_gap": item.score_gap,
            "selection_distinct_score_gap": item.distinct_score_gap,
            "selection_best_score": item.best_score,
            "selection_best_tie_count": item.best_tie_count,
            "selection_best_action_se": item.best_action_se,
            "selection_max_action_se": item.max_action_se,
            "selection_max_model_regret": item.max_model_regret,
            "selection_model_metrics": item.model_metrics,
            "source_bucket": item.selection_reasons[0] if item.selection_reasons else "unknown",
            "source_bucket_group": "turn1_refinement",
        }
    )
    return record


def summary(scored: list[ScoredRecord], targets: list[ScoredRecord], args: argparse.Namespace) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    seat_counts: dict[str, int] = {}
    for item in targets:
        seat_counts[item.seat] = seat_counts.get(item.seat, 0) + 1
        for reason in item.selection_reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    regrets = [item.max_model_regret for item in targets]
    gaps = [item.score_gap for item in targets]
    max_se_values = [item.max_action_se for item in targets]
    return {
        "schema": "hu_turn1_stage1_refinement_target_summary_v1",
        "input": str(args.input),
        "output": str(args.output),
        "records": len(scored),
        "targets": len(targets),
        "self_model": str(args.self_model) if args.self_model else None,
        "hu_model": str(args.hu_model) if args.hu_model else None,
        "reason_counts": dict(sorted(reason_counts.items())),
        "seat_counts": dict(sorted(seat_counts.items())),
        "max_model_regret_mean": float(np.mean(regrets)) if regrets else 0.0,
        "max_model_regret_p90": float(np.quantile(regrets, 0.9)) if regrets else 0.0,
        "max_action_se_mean": float(np.mean(max_se_values)) if max_se_values else 0.0,
        "max_action_se_p90": float(np.quantile(max_se_values, 0.9)) if max_se_values else 0.0,
        "high_se_threshold": args.high_se_threshold,
        "score_gap_mean": float(np.mean(gaps)) if gaps else 0.0,
        "score_gap_p90": float(np.quantile(gaps, 0.9)) if gaps else 0.0,
        "seed": args.seed,
    }


def write_metrics(path: Path, targets: list[ScoredRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in targets:
        rows.append(
            {
                "target_id": len(rows),
                "source_index": item.index,
                "seat": item.seat,
                "selection_reasons": ";".join(item.selection_reasons),
                "score_gap": item.score_gap,
                "distinct_score_gap": item.distinct_score_gap,
                "best_tie_count": item.best_tie_count,
                "best_action_se": item.best_action_se,
                "max_action_se": item.max_action_se,
                "max_model_regret": item.max_model_regret,
                "self_regret": item.model_metrics.get("self_model", {}).get("regret"),
                "hu_regret": item.model_metrics.get("hu_model", {}).get("regret"),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["target_id"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.max_targets <= 0:
        raise SystemExit("--max-targets must be positive")
    records = read_jsonl(args.input)
    scored = score_records(records, self_model_path=args.self_model, hu_model_path=args.hu_model)
    targets = select_targets(
        scored,
        high_regret_count=args.high_regret_count,
        high_se_count=args.high_se_count,
        high_se_threshold=args.high_se_threshold,
        high_gap_count=args.high_gap_count,
        low_gap_count=args.low_gap_count,
        random_count=args.random_count,
        low_gap_threshold=args.low_gap_threshold,
        max_targets=args.max_targets,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for target_id, item in enumerate(targets):
            handle.write(json.dumps(output_record(item, target_id), ensure_ascii=False, separators=(",", ":")) + "\n")
    result = summary(scored, targets, args)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.metrics_output:
        write_metrics(args.metrics_output, targets)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
