"""Train/evaluate a lightweight T2 selector over refined runtime candidates.

This is an offline probe for replacing the hand-tuned
``refined_score + weight * model_score`` selection rule.  It trains a ridge
linear model to predict exact/cap50 EV for refined candidates and reports
leave-one-set-out behavior before any runtime promotion.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ai.training.evaluate_t2_selection_weight import (
    action_key,
    candidate_action_key,
    candidate_selection_score,
    load_extra_scores,
    read_jsonl,
    runtime_result,
    teacher_exact_scores,
)


FEATURES = [
    "bias",
    "refined_score",
    "model_score",
    "refined_plus_model_15",
    "predicted_bust",
    "predicted_fl",
    "predicted_aa",
    "predicted_kk",
    "predicted_qq",
    "predicted_trips",
    "model_rank",
    "inv_model_rank",
    "refined_rank",
    "inv_refined_rank",
    "samples",
    "forced_bust",
    "place_top",
    "place_mid",
    "place_bot",
    "top_len",
    "mid_len",
    "bot_len",
]


@dataclass(frozen=True)
class CandidateRow:
    set_name: str
    row_id: int
    action_key: tuple[tuple[tuple[str, str], ...], str]
    exact_score: float | None
    features: dict[str, float]


@dataclass(frozen=True)
class Group:
    set_name: str
    row_id: int
    best_score: float
    candidates: list[CandidateRow]


def parse_extra_spec(raw: str) -> tuple[Path, set[str]]:
    path, sep, datasets = raw.partition("#")
    dataset_filter = {part for part in datasets.split(",") if part} if sep else set()
    return Path(path), dataset_filter


def parse_eval(raw: str) -> tuple[str, Path, Path, list[tuple[Path, set[str]]]]:
    parts = raw.split("|")
    if len(parts) < 3:
        raise ValueError("--eval must be name|teacher|runtime|extra1;extra2")
    extras = [parse_extra_spec(part) for part in parts[3].split(";") if part.strip()] if len(parts) >= 4 else []
    return parts[0], Path(parts[1]), Path(parts[2]), extras


def rank_value(card: str | None) -> float:
    if not card:
        return 0.0
    rank = str(card)[0]
    values = {
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "T": 10,
        "J": 11,
        "Q": 12,
        "K": 13,
        "A": 14,
        "X": 15,
    }
    return float(values.get(rank, 0))


def candidate_features(candidate: dict[str, Any]) -> dict[str, float]:
    refined = float(candidate.get("refined_score", 0.0) or 0.0)
    model = float(candidate.get("model_score", 0.0) or 0.0)
    model_rank = float(candidate.get("model_rank", 999.0) or 999.0)
    refined_rank = float(candidate.get("refined_rank", 999.0) or 999.0)
    fl_types = candidate.get("predicted_fl_types") or {}
    action = candidate.get("action") or {}
    placements = action.get("placements") or []
    row_counts = {"top": 0, "middle": 0, "mid": 0, "bottom": 0, "bot": 0}
    for _, row in placements:
        row_counts[str(row)] = row_counts.get(str(row), 0) + 1
    board = candidate.get("board") or {}
    top = board.get("top") or []
    mid = board.get("middle", board.get("mid", [])) or []
    bot = board.get("bottom", board.get("bot", [])) or []
    discard_rank = rank_value(action.get("discard"))
    return {
        "bias": 1.0,
        "refined_score": refined,
        "model_score": model,
        "refined_plus_model_15": refined + 1.5 * model,
        "predicted_bust": float(candidate.get("predicted_bust", 0.0) or 0.0),
        "predicted_fl": float(candidate.get("predicted_fl", 0.0) or 0.0),
        "predicted_aa": float(fl_types.get("aa", 0.0) or 0.0),
        "predicted_kk": float(fl_types.get("kk", 0.0) or 0.0),
        "predicted_qq": float(fl_types.get("qq", 0.0) or 0.0),
        "predicted_trips": float(fl_types.get("trips", 0.0) or 0.0),
        "model_rank": model_rank,
        "inv_model_rank": 1.0 / max(model_rank, 1.0),
        "refined_rank": refined_rank,
        "inv_refined_rank": 1.0 / max(refined_rank, 1.0),
        "samples": float(candidate.get("samples", 0.0) or 0.0),
        "forced_bust": 1.0 if bool(candidate.get("forced_bust")) else 0.0,
        "place_top": float(row_counts.get("top", 0)),
        "place_mid": float(row_counts.get("middle", 0) + row_counts.get("mid", 0)),
        "place_bot": float(row_counts.get("bottom", 0) + row_counts.get("bot", 0)),
        "top_len": float(len(top)),
        "mid_len": float(len(mid)),
        "bot_len": float(len(bot)),
        "discard_rank": discard_rank,
    }


def vector(features: dict[str, float]) -> np.ndarray:
    return np.asarray([float(features.get(name, 0.0)) for name in FEATURES], dtype=np.float64)


def build_groups(eval_specs: list[tuple[str, Path, Path, list[tuple[Path, set[str]]]]]) -> list[Group]:
    groups: list[Group] = []
    for set_name, teacher_path, runtime_path, extra_specs in eval_specs:
        teacher_rows = read_jsonl(teacher_path)
        runtime_rows = read_jsonl(runtime_path)
        extras = load_extra_scores(extra_specs)
        for row_id, (teacher, raw_runtime) in enumerate(zip(teacher_rows, runtime_rows)):
            runtime = runtime_result(raw_runtime)
            scores = teacher_exact_scores(teacher, row_id, extras)
            if not scores:
                continue
            best_score = max(scores.values())
            candidates: list[CandidateRow] = []
            for candidate in runtime.get("candidates") or []:
                if not isinstance(candidate, dict) or candidate.get("refined_score") is None:
                    continue
                key = candidate_action_key(candidate)
                candidates.append(
                    CandidateRow(
                        set_name=set_name,
                        row_id=row_id,
                        action_key=key,
                        exact_score=scores.get(key),
                        features=candidate_features(candidate),
                    )
                )
            if candidates:
                groups.append(Group(set_name=set_name, row_id=row_id, best_score=best_score, candidates=candidates))
    return groups


def fit_ridge(groups: list[Group], *, l2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [candidate for group in groups for candidate in group.candidates if candidate.exact_score is not None]
    if len(rows) < len(FEATURES):
        raise ValueError("not enough labeled candidate rows")
    x = np.vstack([vector(row.features) for row in rows])
    y = np.asarray([float(row.exact_score) for row in rows], dtype=np.float64)
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    means[0] = 0.0
    scales[0] = 1.0
    scales[scales < 1e-9] = 1.0
    xs = (x - means) / scales
    reg = np.eye(xs.shape[1]) * float(l2)
    reg[0, 0] = 0.0
    weights = np.linalg.solve(xs.T @ xs + reg, xs.T @ y)
    return weights, means, scales


def selector_score(candidate: CandidateRow, weights: np.ndarray, means: np.ndarray, scales: np.ndarray) -> float:
    x = (vector(candidate.features) - means) / scales
    return float(x @ weights)


def summarize_choice(groups: list[Group], chooser) -> dict[str, Any]:
    rows = known = missing = hits = 0
    regret = 0.0
    for group in groups:
        if not group.candidates:
            continue
        chosen = chooser(group.candidates)
        rows += 1
        if chosen.exact_score is None:
            missing += 1
            continue
        known += 1
        row_regret = max(0.0, group.best_score - float(chosen.exact_score))
        regret += row_regret
        if row_regret <= 1e-9:
            hits += 1
    return {
        "rows": rows,
        "known": known,
        "missing": missing,
        "hits": hits,
        "hit_rate_known": hits / max(known, 1),
        "avg_regret_known": regret / max(known, 1),
    }


def known_oracle(groups: list[Group]) -> dict[str, Any]:
    return summarize_choice(
        groups,
        lambda candidates: max(
            (candidate for candidate in candidates if candidate.exact_score is not None),
            key=lambda candidate: float(candidate.exact_score),
            default=candidates[0],
        ),
    )


def baseline_weight(groups: list[Group], weight: float) -> dict[str, Any]:
    return summarize_choice(
        groups,
        lambda candidates: max(
            candidates,
            key=lambda candidate: (
                candidate.features["refined_score"] + weight * candidate.features["model_score"],
                candidate.features["refined_score"],
                candidate.features["model_score"],
                -candidate.features["model_rank"],
            ),
        ),
    )


def model_choice(groups: list[Group], weights: np.ndarray, means: np.ndarray, scales: np.ndarray) -> dict[str, Any]:
    return summarize_choice(
        groups,
        lambda candidates: max(candidates, key=lambda candidate: selector_score(candidate, weights, means, scales)),
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", action="append", required=True, help="name|teacher|runtime|extra1;extra2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--l2", type=float, default=10.0)
    args = parser.parse_args(argv)

    eval_specs = [parse_eval(raw) for raw in args.eval]
    groups = build_groups(eval_specs)
    set_names = sorted({group.set_name for group in groups})
    all_weights, all_means, all_scales = fit_ridge(groups, l2=args.l2)
    output: dict[str, Any] = {
        "evals": list(args.eval),
        "features": FEATURES,
        "l2": float(args.l2),
        "rows": len(groups),
        "candidate_rows": sum(len(group.candidates) for group in groups),
        "labeled_candidate_rows": sum(
            1 for group in groups for candidate in group.candidates if candidate.exact_score is not None
        ),
        "all": {
            "baseline_weight_1_0": baseline_weight(groups, 1.0),
            "baseline_weight_1_5": baseline_weight(groups, 1.5),
            "baseline_weight_2_0": baseline_weight(groups, 2.0),
            "linear_selector_train_eval": model_choice(groups, all_weights, all_means, all_scales),
            "known_oracle": known_oracle(groups),
        },
        "loso": {},
    }
    for heldout in set_names:
        train = [group for group in groups if group.set_name != heldout]
        test = [group for group in groups if group.set_name == heldout]
        weights, means, scales = fit_ridge(train, l2=args.l2)
        output["loso"][heldout] = {
            "train_rows": len(train),
            "test_rows": len(test),
            "baseline_weight_1_5": baseline_weight(test, 1.5),
            "linear_selector": model_choice(test, weights, means, scales),
            "known_oracle": known_oracle(test),
        }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    base = output["all"]["baseline_weight_1_5"]
    model = output["all"]["linear_selector_train_eval"]
    print(
        f"rows={len(groups)} labeled_candidates={output['labeled_candidate_rows']}/"
        f"{output['candidate_rows']} baseline_reg={base['avg_regret_known']:.3f} "
        f"model_train_reg={model['avg_regret_known']:.3f}"
    )
    for name, fold in output["loso"].items():
        base_fold = fold["baseline_weight_1_5"]
        model_fold = fold["linear_selector"]
        print(
            f"loso={name} baseline_reg={base_fold['avg_regret_known']:.3f} "
            f"model_reg={model_fold['avg_regret_known']:.3f} "
            f"model_missing={model_fold['missing']}/{model_fold['rows']}"
        )


if __name__ == "__main__":
    main()
