"""Build cross-fitted HU Turn1 candidate-vs-baseline selector targets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .hu_turn3_model import read_teacher_samples, sample_to_matrix
from .train_hu_turn1_candidate_generator import build_training_matrix, fit_estimator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026071703)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2", type=float, default=1.0)
    return parser.parse_args()


def stable_fold(row: dict[str, Any], *, folds: int, seed: int) -> int:
    if folds < 2:
        raise ValueError("folds must be >= 2")
    identity = row.get("state_key") or "|".join(
        str(row.get(key, "")) for key in ("hand_seed", "seat", "board", "dealt")
    )
    digest = hashlib.sha1(f"{seed}|{identity}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _training_args() -> SimpleNamespace:
    return SimpleNamespace(
        accept_regret=0.25,
        gray_regret=2.0,
        positive_weight=4.0,
        gray_weight=0.75,
        negative_weight=1.0,
        hard_negative_regret=10.0,
        hard_negative_weight=3.0,
    )


def _action_index(action: dict[str, Any], row_index: int) -> int:
    return int(action.get("original_index", action.get("action_index", row_index)))


def build_target_row(
    row: dict[str, Any],
    *,
    candidate_row_index: int,
    predictions: np.ndarray,
    fold: int,
) -> dict[str, Any]:
    actions = row["actions"]
    baseline_row_index = int(row["baseline_action_row_index"])
    candidate = actions[candidate_row_index]
    baseline = actions[baseline_row_index]
    candidate_ev = float(candidate.get("score", candidate.get("ev", 0.0)))
    baseline_ev = float(baseline.get("score", baseline.get("ev", 0.0)))
    best_ev = max(float(action.get("score", action.get("ev", 0.0))) for action in actions)
    output = dict(row)
    output.update(
        {
            "schema": "hu_turn1_oof_selector_target_v1",
            "oof_fold": int(fold),
            "oof_prediction": True,
            "override_fired": True,
            "hu_turn1_action": candidate,
            "candidate_action_index": _action_index(candidate, candidate_row_index),
            "candidate_action_row_index": int(candidate_row_index),
            "fallback_action": baseline,
            "fallback_action_index": _action_index(baseline, baseline_row_index),
            "candidate_score": float(predictions[candidate_row_index]),
            "fallback_score": float(predictions[baseline_row_index]),
            "hu_turn1_predicted_margin": float(
                predictions[candidate_row_index] - predictions[baseline_row_index]
            ),
            "stage10_mc32_candidate_delta": float(candidate_ev - baseline_ev),
            "teacher_candidate_ev": candidate_ev,
            "teacher_baseline_ev": baseline_ev,
            "teacher_best_ev": best_ev,
            "teacher_candidate_regret": float(best_ev - candidate_ev),
            "realized_delta_valid": False,
            "selector_label_basis": "oof_candidate_mc_teacher_delta",
            "relabel_compare": {
                "source_best_new_regret": float(best_ev - candidate_ev),
                "best_action_changed": candidate_row_index != int(row.get("best_action", 0)),
            },
        }
    )
    return output


def build_oof_targets(
    rows: Sequence[dict[str, Any]],
    *,
    folds: int,
    seed: int,
    max_iter: int,
    learning_rate: float,
    max_leaf_nodes: int,
    l2: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fold_ids = np.asarray([stable_fold(row, folds=folds, seed=seed) for row in rows], dtype=np.int64)
    targets: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []
    same_as_baseline = 0
    for fold in range(folds):
        train_rows = [row for index, row in enumerate(rows) if fold_ids[index] != fold]
        heldout_rows = [row for index, row in enumerate(rows) if fold_ids[index] == fold]
        if not train_rows or not heldout_rows:
            raise ValueError(f"fold {fold} has an empty train or heldout split")
        train_x, _labels, weights, _offsets, _regrets, scores = build_training_matrix(
            train_rows,
            source_weights={},
            args=_training_args(),
            include_scores=True,
        )
        estimator = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=l2,
            random_state=seed + fold,
            early_stopping=True,
        )
        fit_estimator(estimator, train_x, scores, weights)
        regrets: list[float] = []
        deltas: list[float] = []
        fold_same = 0
        for row in heldout_rows:
            features, teacher_scores = sample_to_matrix(row)
            predictions = np.asarray(estimator.predict(features), dtype=np.float64)
            candidate_row_index = int(np.argmax(predictions))
            target = build_target_row(
                row,
                candidate_row_index=candidate_row_index,
                predictions=predictions,
                fold=fold,
            )
            if candidate_row_index == int(row["baseline_action_row_index"]):
                same_as_baseline += 1
                fold_same += 1
                continue
            targets.append(target)
            regrets.append(float(np.max(teacher_scores) - teacher_scores[candidate_row_index]))
            deltas.append(float(target["stage10_mc32_candidate_delta"]))
        fold_metrics.append(
            {
                "fold": fold,
                "train_rows": len(train_rows),
                "heldout_rows": len(heldout_rows),
                "candidate_diff_rows": len(deltas),
                "same_as_baseline_rows": fold_same,
                "candidate_avg_regret": float(np.mean(regrets)) if regrets else 0.0,
                "candidate_delta_vs_baseline_mean": float(np.mean(deltas)) if deltas else 0.0,
                "candidate_positive_rate": float(np.mean(np.asarray(deltas) > 0.0)) if deltas else 0.0,
            }
        )
    all_deltas = np.asarray([float(row["stage10_mc32_candidate_delta"]) for row in targets])
    summary = {
        "schema": "hu_turn1_oof_selector_target_summary_v1",
        "input_rows": len(rows),
        "folds": folds,
        "seed": seed,
        "output_rows": len(targets),
        "same_as_baseline_rows": same_as_baseline,
        "positive_rows": int(np.sum(all_deltas > 0.0)),
        "negative_rows": int(np.sum(all_deltas < 0.0)),
        "zero_rows": int(np.sum(all_deltas == 0.0)),
        "candidate_delta_mean": float(np.mean(all_deltas)) if all_deltas.size else 0.0,
        "fold_metrics": fold_metrics,
    }
    return targets, summary


def main() -> None:
    args = parse_args()
    rows = read_teacher_samples(args.input)
    if args.folds < 2:
        raise SystemExit("--folds must be >= 2")
    targets, summary = build_oof_targets(
        rows,
        folds=int(args.folds),
        seed=int(args.seed),
        max_iter=int(args.max_iter),
        learning_rate=float(args.learning_rate),
        max_leaf_nodes=int(args.max_leaf_nodes),
        l2=float(args.l2),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in targets:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
