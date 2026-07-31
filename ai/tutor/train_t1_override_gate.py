"""Train a small T1 override gate from runtime refinement rows.

The gate predicts whether a T1 recursive-refinement override should replace the
model Top1.  Labels come from stronger teacher records, not from the noisy
runtime MC300 teacher used to discover targets.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.action_space import encode_action, get_turn_actions
from ai.tutor.evaluate_hybrid_refinement_teacher import canonical_action, candidate_action, candidate_score
from ai.tutor.hybrid_t1t2 import action_to_dict, enrich_override_gate_features, observation_from_payload


BASE_FEATURES = [
    "refined_delta",
    "model_score_delta",
    "model_raw_score_delta",
    "model_rank_gap",
    "predicted_bust_delta",
    "predicted_fl_delta",
    "predicted_qq_delta",
    "predicted_kk_delta",
    "predicted_aa_delta",
    "predicted_trips_delta",
    "best_refined_score",
    "model_refined_score",
    "best_model_score",
    "model_top1_score",
    "best_predicted_bust",
    "model_predicted_bust",
    "best_predicted_fl",
    "model_predicted_fl",
]

DERIVED_FEATURES = [
    "abs_refined_delta",
    "abs_model_score_delta",
    "refined_delta_per_rank_gap",
    "refined_delta_minus_abs_model_score_delta",
    "refined_delta_plus_model_score_delta",
    "refined_delta_x_rank_gap",
    "refined_delta_x_model_score_delta",
    "refined_delta_x_predicted_bust_delta",
    "refined_delta_x_predicted_fl_delta",
    "rank_gap_is_one",
    "rank_gap_is_two_plus",
    "best_refined_minus_best_model_score",
    "model_refined_minus_model_top1_score",
    "fl_type_delta_sum",
    "premium_fl_delta_sum",
    "risk_adjusted_refined_delta",
    "fl_adjusted_refined_delta",
]

DEFAULT_FEATURES = BASE_FEATURES + DERIVED_FEATURES


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def state_key(record: dict[str, Any]) -> str:
    return json.dumps(
        {
            "turn": record.get("turn"),
            "board": record.get("board"),
            "opponent_board": record.get("opponent_board"),
            "dealt": record.get("dealt"),
            "known_discards": record.get("known_discards"),
            "is_btn": record.get("is_btn"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def action_map(record: dict[str, Any]) -> dict[int, tuple[tuple[tuple[str, str], ...], str]]:
    obs = observation_from_payload(record)
    valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    return {
        int(encode_action(action, valid_actions, turn=obs.turn, dealt_cards=obs.dealt_cards)): canonical_action(
            action_to_dict(action)
        )
        for action in valid_actions
    }


def teacher_scores(record: dict[str, Any]) -> dict[tuple[tuple[tuple[str, str], ...], str], float]:
    return {
        canonical_action(candidate_action(candidate)): candidate_score(candidate)
        for candidate in (record.get("candidates") or [])
    }


def load_runtime_input(path: Path) -> dict[int, dict[str, Any]]:
    return {line_no: row for line_no, row in enumerate(iter_jsonl(path), start=1)}


def load_strong_teachers(paths: list[Path]) -> dict[str, dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in iter_jsonl(path):
            by_key.setdefault(state_key(row), row)
    return by_key


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    runtime_input = load_runtime_input(Path(args.runtime_input))
    teachers = load_strong_teachers([Path(part) for part in args.strong_teachers.split(";") if part.strip()])
    rows: list[dict[str, Any]] = []
    stats = Counter()
    for result in iter_jsonl(Path(args.runtime_results)):
        stats["runtime_rows"] += 1
        if int(result.get("turn", -1)) != 1:
            stats["skipped_turn"] += 1
            continue
        features = result.get("t1_override_features")
        if not isinstance(features, dict):
            stats["skipped_no_features"] += 1
            continue
        if not result.get("model_top1_action_idx") or not result.get("best_action_idx"):
            stats["skipped_missing_action"] += 1
            continue
        source_record = runtime_input.get(int(result.get("line", 0)))
        if source_record is None:
            stats["skipped_missing_runtime_input"] += 1
            continue
        teacher = teachers.get(state_key(source_record))
        if teacher is None:
            stats["skipped_missing_strong_teacher"] += 1
            continue
        idx_to_action = action_map(source_record)
        scores = teacher_scores(teacher)
        model_action = idx_to_action.get(int(result["model_top1_action_idx"]))
        final_action = idx_to_action.get(int(result["best_action_idx"]))
        model_score = scores.get(model_action)
        final_score = scores.get(final_action)
        if model_score is None or final_score is None:
            stats["skipped_unmatched_action"] += 1
            continue
        best_action = max(scores.items(), key=lambda item: item[1])[0] if scores else None
        teacher_best_score = scores.get(best_action, max(float(model_score), float(final_score)))
        model_top1_hit = model_action == best_action
        final_top1_hit = final_action == best_action
        improvement = float(final_score) - float(model_score)
        if args.drop_ties and abs(improvement) < args.tie_epsilon:
            stats["skipped_tie"] += 1
            continue
        if args.label_mode == "score":
            label = 1 if improvement > 0.0 else 0
        elif args.label_mode == "top1":
            label = 1 if final_top1_hit else 0
        elif args.label_mode == "top1_delta":
            label = 1 if final_top1_hit and not model_top1_hit else 0
            if final_top1_hit == model_top1_hit:
                stats["skipped_same_top1_status"] += 1
                continue
        else:
            raise ValueError(f"unsupported label mode: {args.label_mode}")
        enriched_features = enrich_override_gate_features(
            {name: float(features.get(name, 0.0)) for name in BASE_FEATURES}
        )
        rows.append(
            {
                "line": result.get("line"),
                "source": source_record.get("source"),
                "source_line": source_record.get("source_line"),
                "features": {name: float(enriched_features.get(name, 0.0)) for name in DEFAULT_FEATURES},
                "label": label,
                "improvement": improvement,
                "model_score": float(model_score),
                "final_score": float(final_score),
                "teacher_best_score": float(teacher_best_score),
                "model_top1_hit": bool(model_top1_hit),
                "final_top1_hit": bool(final_top1_hit),
                "teacher_eval_mode": teacher.get("eval_mode"),
                "runtime_refined_delta": result.get("refined_override_delta"),
            }
        )
        stats["label_accept" if label else "label_reject"] += 1
    if args.rows_output:
        output = Path(args.rows_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.with_suffix(".summary.json").write_text(
            json.dumps({"rows": len(rows), "stats": dict(stats)}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return rows


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        features = enrich_override_gate_features(
            {name: float((row.get("features") or {}).get(name, 0.0)) for name in BASE_FEATURES}
        )
        out = dict(row)
        out["features"] = {name: float(features.get(name, 0.0)) for name in DEFAULT_FEATURES}
        out["label"] = int(out.get("label", 0))
        out["improvement"] = float(out.get("improvement", 0.0))
        rows.append(out)
    return rows


def load_rows_inputs(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any]] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        for row in load_rows(path):
            key = (row.get("source"), row.get("source_line"), row.get("line"))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def standardize(rows: list[dict[str, Any]], feature_names: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in feature_names:
        values = [float(row["features"].get(name, 0.0)) for row in rows]
        means[name] = statistics.fmean(values) if values else 0.0
        scales[name] = statistics.pstdev(values) if len(values) > 1 else 1.0
        if scales[name] < 1e-9:
            scales[name] = 1.0
    return means, scales


def sigmoid(value: float) -> float:
    if value >= 0:
        ez = math.exp(-value)
        return 1.0 / (1.0 + ez)
    ez = math.exp(value)
    return ez / (1.0 + ez)


def train_logistic(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    *,
    lr: float,
    l2: float,
    epochs: int,
) -> tuple[list[float], float, dict[str, float], dict[str, float]]:
    means, scales = standardize(rows, feature_names)
    weights = [0.0 for _ in feature_names]
    intercept = 0.0
    n = max(len(rows), 1)
    for _ in range(epochs):
        grad = [0.0 for _ in feature_names]
        grad_b = 0.0
        for row in rows:
            z = intercept
            for i, name in enumerate(feature_names):
                x = (float(row["features"].get(name, 0.0)) - means[name]) / scales[name]
                z += weights[i] * x
            pred = sigmoid(z)
            err = pred - float(row["label"])
            grad_b += err
            for i, name in enumerate(feature_names):
                x = (float(row["features"].get(name, 0.0)) - means[name]) / scales[name]
                grad[i] += err * x
        intercept -= lr * grad_b / n
        for i in range(len(weights)):
            weights[i] -= lr * ((grad[i] / n) + l2 * weights[i])
    return weights, intercept, means, scales


def predict(row: dict[str, Any], feature_names: list[str], weights: list[float], intercept: float, means, scales) -> float:
    z = intercept
    for i, name in enumerate(feature_names):
        x = (float(row["features"].get(name, 0.0)) - means[name]) / scales[name]
        z += weights[i] * x
    return sigmoid(z)


def evaluate_threshold(rows: list[dict[str, Any]], probs: list[float], threshold: float) -> dict[str, Any]:
    accepted = [prob >= threshold for prob in probs]
    tp = sum(1 for row, acc in zip(rows, accepted) if acc and row["label"] == 1)
    fp = sum(1 for row, acc in zip(rows, accepted) if acc and row["label"] == 0)
    tn = sum(1 for row, acc in zip(rows, accepted) if not acc and row["label"] == 0)
    fn = sum(1 for row, acc in zip(rows, accepted) if not acc and row["label"] == 1)
    accepted_count = sum(1 for acc in accepted if acc)
    rejected_count = sum(1 for acc in accepted if not acc)
    total_gain = sum(float(row["improvement"]) for row, acc in zip(rows, accepted) if acc)
    model_baseline_hits = sum(1 for row in rows if bool(row.get("model_top1_hit")))
    final_if_always_accept_hits = sum(1 for row in rows if bool(row.get("final_top1_hit")))
    gated_hits = sum(
        1
        for row, acc in zip(rows, accepted)
        if (bool(row.get("final_top1_hit")) if acc else bool(row.get("model_top1_hit")))
    )
    accepted_final_hits = sum(1 for row, acc in zip(rows, accepted) if acc and bool(row.get("final_top1_hit")))
    rejected_model_hits = sum(1 for row, acc in zip(rows, accepted) if not acc and bool(row.get("model_top1_hit")))
    accepted_bad_top1 = sum(
        1
        for row, acc in zip(rows, accepted)
        if acc and bool(row.get("model_top1_hit")) and not bool(row.get("final_top1_hit"))
    )
    rejected_missed_top1 = sum(
        1
        for row, acc in zip(rows, accepted)
        if not acc and bool(row.get("final_top1_hit")) and not bool(row.get("model_top1_hit"))
    )
    selected_regret_sum = 0.0
    model_regret_sum = 0.0
    has_best_scores = False
    for row, acc in zip(rows, accepted):
        if "teacher_best_score" not in row:
            continue
        has_best_scores = True
        best_score = float(row.get("teacher_best_score", 0.0))
        model_score = float(row.get("model_score", 0.0))
        final_score = float(row.get("final_score", 0.0))
        selected_score = final_score if acc else model_score
        selected_regret_sum += max(0.0, best_score - selected_score)
        model_regret_sum += max(0.0, best_score - model_score)
    return {
        "threshold": threshold,
        "accepted": accepted_count,
        "rejected": rejected_count,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / max(len(rows), 1),
        "accepted_gain": total_gain,
        "avg_accepted_gain": total_gain / max(accepted_count, 1),
        "model_baseline_top1": model_baseline_hits,
        "final_if_always_accept_top1": final_if_always_accept_hits,
        "gated_top1": gated_hits,
        "gated_top1_recall": gated_hits / max(len(rows), 1),
        "top1_delta_vs_model": (gated_hits - model_baseline_hits) / max(len(rows), 1),
        "top1_delta_vs_always_accept": (gated_hits - final_if_always_accept_hits) / max(len(rows), 1),
        "accepted_final_top1": accepted_final_hits,
        "rejected_model_top1": rejected_model_hits,
        "accepted_bad_top1": accepted_bad_top1,
        "rejected_missed_top1": rejected_missed_top1,
        "selected_regret": selected_regret_sum if has_best_scores else None,
        "model_baseline_regret": model_regret_sum if has_best_scores else None,
        "selected_avg_regret": (selected_regret_sum / max(len(rows), 1)) if has_best_scores else None,
        "model_baseline_avg_regret": (model_regret_sum / max(len(rows), 1)) if has_best_scores else None,
    }


def threshold_sort_key(row: dict[str, Any], objective: str) -> tuple[Any, ...]:
    if objective == "accepted_gain":
        return (row["accepted_gain"], row["accuracy"], -row["fp"])
    if objective == "top1":
        return (
            row["gated_top1"],
            -row["accepted_bad_top1"],
            -row["rejected_missed_top1"],
            row["accepted_gain"],
            row["accuracy"],
        )
    if objective == "top1_then_regret":
        selected_regret = row.get("selected_regret")
        regret_key = -float(selected_regret) if selected_regret is not None else 0.0
        return (
            row["gated_top1"],
            regret_key,
            -row["accepted_bad_top1"],
            -row["rejected_missed_top1"],
            row["accepted_gain"],
        )
    raise ValueError(f"unsupported threshold objective: {objective}")


def threshold_grid(step: float) -> list[float]:
    step = max(float(step), 0.001)
    count = int(round(1.0 / step))
    values = sorted({round(i * step, 10) for i in range(0, count + 1)})
    return [value for value in values if 0.0 <= value <= 1.0]


def train(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if args.rows_input:
        rows.extend(load_rows_inputs([Path(part) for part in args.rows_input.split(";") if part.strip()]))
    if args.runtime_results:
        rows.extend(build_rows(args))
    if not rows:
        raise SystemExit("no training rows")
    feature_names = [part for part in args.features.split(",") if part.strip()]
    weights, intercept, means, scales = train_logistic(
        rows,
        feature_names,
        lr=args.lr,
        l2=args.l2,
        epochs=args.epochs,
    )
    probs = [predict(row, feature_names, weights, intercept, means, scales) for row in rows]
    thresholds = threshold_grid(args.threshold_step)
    threshold_rows = [evaluate_threshold(rows, probs, threshold) for threshold in thresholds]
    best = max(
        threshold_rows,
        key=lambda row: threshold_sort_key(row, args.threshold_objective),
    )
    gate = {
        "kind": "linear_logistic",
        "features": feature_names,
        "weights": weights,
        "intercept": intercept,
        "means": means,
        "scales": scales,
        "recommended_threshold": best["threshold"],
        "training_rows": len(rows),
        "labels": {
            "accept": sum(1 for row in rows if row["label"] == 1),
            "reject": sum(1 for row in rows if row["label"] == 0),
        },
        "threshold_eval": threshold_rows,
        "best_threshold_eval": best,
        "source": {
            "runtime_results": args.runtime_results,
            "runtime_input": args.runtime_input,
            "strong_teachers": args.strong_teachers,
            "rows_input": args.rows_input,
            "label_mode": args.label_mode,
            "threshold_objective": args.threshold_objective,
            "threshold_step": args.threshold_step,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")
    return gate


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a T1 override gate from stronger relabeled teacher rows")
    parser.add_argument("--runtime-results", default="")
    parser.add_argument("--runtime-input", default="")
    parser.add_argument("--strong-teachers", default="", help="Semicolon-separated teacher JSONL paths")
    parser.add_argument("--rows-input", default="", help="Semicolon-separated prebuilt gate row JSONL paths")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows-output", default="")
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--label-mode", choices=["score", "top1", "top1_delta"], default="score")
    parser.add_argument(
        "--threshold-objective",
        choices=["accepted_gain", "top1", "top1_then_regret"],
        default="accepted_gain",
    )
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--drop-ties", action="store_true")
    parser.add_argument("--tie-epsilon", type=float, default=1e-9)
    args = parser.parse_args(list(argv) if argv is not None else None)
    gate = train(args)
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
