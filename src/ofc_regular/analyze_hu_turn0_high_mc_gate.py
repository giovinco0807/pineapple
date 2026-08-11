"""Calibrate deployable HU T0 score-margin gates on high-MC holdout rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Protocol, Sequence

from .action_space import Action
from .hu_turn0_candidate import load_turn0_candidate_model
from .hu_turn3_model import hu_policy_sample
from .state import Board


class CandidateModel(Protocol):
    def predict_sample(self, sample: dict[str, Any]) -> Sequence[float]: ...


def _board(data: dict[str, Any]) -> Board:
    return Board.from_rows(data.get("top", ()), data.get("middle", ()), data.get("bottom", ()))


def _action(data: dict[str, Any]) -> Action:
    return Action(
        placements=tuple((str(card), str(row)) for card, row in data.get("placements", ())),
        discards=tuple(str(card) for card in data.get("discards", ())),
    )


def _original_index(data: dict[str, Any]) -> int:
    return int(data.get("original_index", data.get("action_index")))


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _mean_ci95(values: Sequence[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    center = mean(values)
    if len(values) < 2:
        return center, center, center
    half = 1.96 * stdev(values) / math.sqrt(len(values))
    return center, center - half, center + half


def decision_rows_for_model(
    rows: Sequence[dict[str, Any]],
    model: CandidateModel,
    *,
    model_name: str,
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for row in rows:
        actions_json = list(row.get("actions", ()))
        actions = [_action(action) for action in actions_json]
        sample = hu_policy_sample(
            _board(row["board"]),
            row["dealt"],
            actions,
            opponent_board=_board(row["opponent_board"]),
            dead_cards=tuple(row.get("visible_dead_cards") or row.get("dead_cards") or ()),
            seat=row.get("seat"),
            to_act_order=row.get("to_act_order") or row.get("seat"),
        )
        predictions = [float(value) for value in model.predict_sample(sample)]
        if len(predictions) != len(actions_json):
            raise ValueError(
                f"prediction count mismatch for {model_name}: {len(predictions)} != {len(actions_json)}"
            )
        if not all(math.isfinite(value) for value in predictions):
            raise ValueError(f"non-finite prediction from {model_name}")
        original_to_offset = {
            _original_index(action): offset for offset, action in enumerate(actions_json)
        }
        baseline_original = int(row["baseline_action_index"])
        if baseline_original not in original_to_offset:
            raise ValueError("baseline action is absent from high-MC candidate rows")
        baseline_offset = original_to_offset[baseline_original]
        candidate_offset = max(
            range(len(predictions)),
            key=lambda offset: (predictions[offset], -_original_index(actions_json[offset])),
        )
        candidate = actions_json[candidate_offset]
        teacher_delta = float(candidate["delta_vs_baseline"])
        teacher_delta_se = float(candidate["delta_se_vs_baseline"])
        decisions.append(
            {
                "model": model_name,
                "dataset_state_id": row.get("dataset_state_id"),
                "sample_id": row.get("sample_id"),
                "hand_seed": row.get("hand_seed"),
                "seat": row.get("seat"),
                "candidate_original_index": _original_index(candidate),
                "baseline_original_index": baseline_original,
                "same_as_baseline": candidate_offset == baseline_offset,
                "predicted_delta": predictions[candidate_offset] - predictions[baseline_offset],
                "predicted_candidate_score": predictions[candidate_offset],
                "predicted_baseline_score": predictions[baseline_offset],
                "teacher_delta": teacher_delta,
                "teacher_delta_se": teacher_delta_se,
                "teacher_delta_lcb196": teacher_delta - 1.96 * teacher_delta_se,
                "teacher_delta_z": (
                    teacher_delta / teacher_delta_se
                    if teacher_delta_se > 0.0
                    else (math.inf if teacher_delta > 0.0 else 0.0)
                ),
                "teacher_best_delta": float(row["delta_best_vs_baseline"]),
                "candidate_is_teacher_best": (
                    _original_index(candidate) == int(row["best_action_index"])
                ),
            }
        )
    return decisions


def sweep_gate_thresholds(
    decisions: Sequence[dict[str, Any]],
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for threshold in thresholds:
        fired = [
            row
            for row in decisions
            if not bool(row["same_as_baseline"])
            and float(row["predicted_delta"]) >= float(threshold)
        ]
        results.append(
            _gate_result(
                decisions,
                fired,
                threshold=float(threshold),
                threshold_first=float(threshold),
                threshold_second=float(threshold),
                allowed_seats=("first", "second"),
            )
        )
    return results


def _gate_result(
    decisions: Sequence[dict[str, Any]],
    fired: Sequence[dict[str, Any]],
    *,
    threshold: float | None,
    threshold_first: float | None,
    threshold_second: float | None,
    allowed_seats: tuple[str, ...],
) -> dict[str, Any]:
        total = len(decisions)
        gains = [float(row["teacher_delta"]) for row in fired]
        losses = [max(0.0, -gain) for gain in gains]
        gain_mean, ci_low, ci_high = _mean_ci95(gains)
        first = [row for row in fired if row.get("seat") == "first"]
        second = [row for row in fired if row.get("seat") == "second"]
        return {
            "model": decisions[0]["model"] if decisions else "unknown",
            "threshold": threshold,
            "threshold_first": threshold_first,
            "threshold_second": threshold_second,
            "allowed_seats": ",".join(allowed_seats),
            "rows": total,
            "fires": len(fired),
            "fire_rate": len(fired) / total if total else 0.0,
            "avg_gain": gain_mean,
            "median_gain": median(gains) if gains else 0.0,
            "gain_ci95_low": ci_low,
            "gain_ci95_high": ci_high,
            "estimated_ev_per_state": (len(fired) / total * gain_mean) if total else 0.0,
            "false_positive_count": sum(gain <= 0.0 for gain in gains),
            "false_positive_rate": (
                sum(gain <= 0.0 for gain in gains) / len(gains) if gains else 0.0
            ),
            "lcb196_positive_count": sum(
                float(item["teacher_delta_lcb196"]) > 0.0 for item in fired
            ),
            "lcb196_positive_rate": (
                sum(float(item["teacher_delta_lcb196"]) > 0.0 for item in fired)
                / len(fired)
                if fired
                else 0.0
            ),
            "p95_loss": _quantile(losses, 0.95),
            "p99_loss": _quantile(losses, 0.99),
            "max_loss": max(losses, default=0.0),
            "first_fires": len(first),
            "first_avg_gain": mean(float(item["teacher_delta"]) for item in first)
            if first
            else 0.0,
            "second_fires": len(second),
            "second_avg_gain": mean(float(item["teacher_delta"]) for item in second)
            if second
            else 0.0,
        }


def sweep_seat_thresholds(
    decisions: Sequence[dict[str, Any]],
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for first_threshold in thresholds:
        for second_threshold in thresholds:
            fired = [
                row
                for row in decisions
                if not bool(row["same_as_baseline"])
                and float(row["predicted_delta"])
                >= (
                    float(first_threshold)
                    if row.get("seat") == "first"
                    else float(second_threshold)
                )
            ]
            results.append(
                _gate_result(
                    decisions,
                    fired,
                    threshold=None,
                    threshold_first=float(first_threshold),
                    threshold_second=float(second_threshold),
                    allowed_seats=("first", "second"),
                )
            )
    for seat in ("first", "second"):
        for threshold in thresholds:
            fired = [
                row
                for row in decisions
                if row.get("seat") == seat
                and not bool(row["same_as_baseline"])
                and float(row["predicted_delta"]) >= float(threshold)
            ]
            results.append(
                _gate_result(
                    decisions,
                    fired,
                    threshold=None,
                    threshold_first=float(threshold) if seat == "first" else None,
                    threshold_second=float(threshold) if seat == "second" else None,
                    allowed_seats=(seat,),
                )
            )
    return results


def choose_recommended_gate(results: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not results:
        return None
    total = int(results[0]["rows"])
    min_fires = max(10, int(math.ceil(total * 0.02)))
    eligible = []
    for row in results:
        allowed = set(str(row.get("allowed_seats") or "first,second").split(","))
        seat_coverage = all(int(row[f"{seat}_fires"]) > 0 for seat in allowed)
        if (
            int(row["fires"]) >= min_fires
            and float(row["avg_gain"]) > 0.0
            and float(row["false_positive_rate"]) <= 0.10
            and seat_coverage
        ):
            eligible.append(row)
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (
            float(row["estimated_ev_per_state"]),
            float(row["gain_ci95_low"]),
            -float(row["false_positive_rate"]),
        ),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_thresholds(raw: str) -> list[float]:
    values = sorted({float(part.strip()) for part in raw.split(",") if part.strip()})
    if not values:
        raise ValueError("at least one threshold is required")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--candidate-models", type=Path, nargs="+", required=True)
    parser.add_argument("--thresholds", default="0,0.1,0.25,0.5,0.75,1,1.5,2,2.5,3")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = _read_jsonl(args.input)
    thresholds = parse_thresholds(args.thresholds)
    all_decisions: list[dict[str, Any]] = []
    all_sweeps: list[dict[str, Any]] = []
    all_seat_sweeps: list[dict[str, Any]] = []
    recommendations: list[dict[str, Any]] = []
    seat_recommendations: list[dict[str, Any]] = []
    for path in args.candidate_models:
        model = load_turn0_candidate_model(path)
        decisions = decision_rows_for_model(rows, model, model_name=str(path))
        sweep = sweep_gate_thresholds(decisions, thresholds)
        seat_sweep = sweep_seat_thresholds(decisions, thresholds)
        recommended = choose_recommended_gate(sweep)
        seat_recommended = choose_recommended_gate(seat_sweep)
        all_decisions.extend(decisions)
        all_sweeps.extend(sweep)
        all_seat_sweeps.extend(seat_sweep)
        if recommended is not None:
            recommendations.append(dict(recommended))
        if seat_recommended is not None:
            seat_recommendations.append(dict(seat_recommended))

    overall = choose_recommended_gate([*recommendations, *seat_recommendations])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "gate_sweep.csv", all_sweeps)
    _write_csv(args.output_dir / "seat_gate_sweep.csv", all_seat_sweeps)
    with (args.output_dir / "decision_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_decisions:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "schema": "hu_turn0_high_mc_gate_calibration_v1",
        "input": str(args.input),
        "rows": len(rows),
        "candidate_models": [str(path) for path in args.candidate_models],
        "thresholds": thresholds,
        "per_model_recommendations": recommendations,
        "per_model_seat_recommendations": seat_recommendations,
        "recommended": overall,
        "decision": "fresh_seat_swap_required" if overall else "no_runtime_gate_candidate",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
