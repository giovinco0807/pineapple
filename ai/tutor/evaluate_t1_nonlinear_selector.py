"""Train and audit local nonlinear T1 final selectors from selector rows.

This is offline-only.  It predicts each refined candidate's teacher-relative
score, then chooses the highest predicted candidate within each position.
Runtime promotion should happen only after held-out Top1 and regret improve.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.tutor.train_t1_final_selector import DEFAULT_FEATURES, row_features, selector_score  # noqa: E402


DEFAULT_NONLINEAR_FEATURES = [
    *DEFAULT_FEATURES,
    "final_selector_score",
    "final_selector_minus_refined",
    "final_selector_minus_model",
    "final_selector_x_safe",
    "is_model_top1",
    "is_sync",
    "neg_output_rank",
]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def parse_spec(raw: str) -> DatasetSpec:
    if "=" in raw:
        name, path = raw.split("=", 1)
        return DatasetSpec(name=name.strip(), path=Path(path))
    path = Path(raw)
    return DatasetSpec(name=path.parent.name or path.stem, path=path)


def group_rows(specs: list[DatasetSpec]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in specs:
        for row in iter_jsonl(spec.path):
            if int(row.get("turn", -1)) != 1:
                continue
            if not bool(row.get("is_refined")):
                continue
            if row.get("teacher_score") is None:
                continue
            grouped[f"{spec.name}:{int(row.get('line', 0))}"].append(row)
    return {key: rows for key, rows in grouped.items() if rows}


def fnum(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def teacher_best_score(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return max(fnum(row, "teacher_best_score", fnum(row, "teacher_score", 0.0)) for row in rows)


def row_teacher_score(row: dict[str, Any]) -> float:
    return fnum(row, "teacher_score", fnum(row, "teacher_best_score", 0.0))


def row_target(row: dict[str, Any], best_score: float, target: str) -> float:
    score = row_teacher_score(row)
    regret = max(0.0, best_score - score)
    if target == "neg_regret":
        return -regret
    if target == "score":
        return score
    if target == "zero_one":
        return 1.0 if bool(row.get("is_teacher_best")) else 0.0
    raise ValueError(f"unsupported target: {target}")


def load_json(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def feature_values(row: dict[str, Any], linear_selector: dict[str, Any] | None) -> dict[str, float]:
    values = row_features(row)
    final_score = row.get("final_selector_score")
    if final_score is None and linear_selector is not None:
        final_score = selector_score(row, linear_selector)
    final_selector_score = float(final_score or 0.0)
    refined_score = fnum(row, "refined_score")
    model_score = fnum(row, "model_score")
    predicted_bust = fnum(row, "predicted_bust")
    values.update(
        {
            "final_selector_score": final_selector_score,
            "final_selector_minus_refined": final_selector_score - refined_score,
            "final_selector_minus_model": final_selector_score - model_score,
            "final_selector_x_safe": final_selector_score * (1.0 - predicted_bust),
            "is_model_top1": 1.0 if bool(row.get("is_model_top1")) else 0.0,
            "is_sync": 1.0 if bool(row.get("is_sync")) else 0.0,
            "neg_output_rank": -max(fnum(row, "output_rank", 999.0), 1.0),
        }
    )
    return values


def vector(
    row: dict[str, Any],
    features: list[str],
    linear_selector: dict[str, Any] | None,
) -> list[float]:
    values = feature_values(row, linear_selector)
    out: list[float] = []
    for name in features:
        value = values.get(name, 0.0)
        if value is None or not math.isfinite(float(value)):
            out.append(0.0)
        else:
            out.append(float(value))
    return out


def build_matrix(
    groups: dict[str, list[dict[str, Any]]],
    features: list[str],
    *,
    target: str,
    linear_selector: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_rows: list[list[float]] = []
    y_rows: list[float] = []
    weights: list[float] = []
    for rows in groups.values():
        best = teacher_best_score(rows)
        for row in rows:
            regret = max(0.0, best - row_teacher_score(row))
            x_rows.append(vector(row, features, linear_selector))
            y_rows.append(row_target(row, best, target))
            weights.append(1.0 + min(regret, 5.0))
    return (
        np.asarray(x_rows, dtype=np.float32),
        np.asarray(y_rows, dtype=np.float32),
        np.asarray(weights, dtype=np.float32),
    )


def pick_runtime(rows: list[dict[str, Any]]) -> dict[str, Any]:
    runtime = [row for row in rows if bool(row.get("is_runtime_best"))]
    if runtime:
        return runtime[0]
    return max(rows, key=lambda row: (fnum(row, "final_selector_score"), fnum(row, "refined_score")))


def pick_refined(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: (fnum(row, "refined_score"), fnum(row, "model_score")))


def pick_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: fnum(row, "model_score"))


def summarize_choice(groups: dict[str, list[dict[str, Any]]], picker) -> dict[str, Any]:
    hits = 0
    zero = 0
    regret = 0.0
    max_regret = 0.0
    oracle_hits = 0
    oracle_zero = 0
    oracle_regret = 0.0
    for rows in groups.values():
        best = teacher_best_score(rows)
        chosen = picker(rows)
        chosen_score = row_teacher_score(chosen)
        chosen_regret = max(0.0, best - chosen_score)
        regret += chosen_regret
        max_regret = max(max_regret, chosen_regret)
        if bool(chosen.get("is_teacher_best")):
            hits += 1
        if chosen_regret <= 1e-9:
            zero += 1

        oracle = max(rows, key=row_teacher_score)
        oracle_score = row_teacher_score(oracle)
        oracle_regret_value = max(0.0, best - oracle_score)
        oracle_regret += oracle_regret_value
        if bool(oracle.get("is_teacher_best")):
            oracle_hits += 1
        if oracle_regret_value <= 1e-9:
            oracle_zero += 1
    denom = max(len(groups), 1)
    return {
        "decisions": len(groups),
        "top1": hits / denom,
        "zero_regret": zero / denom,
        "avg_regret": regret / denom,
        "max_regret": max_regret,
        "oracle_refined_top1": oracle_hits / denom,
        "oracle_refined_zero_regret": oracle_zero / denom,
        "oracle_refined_avg_regret": oracle_regret / denom,
    }


def model_picker(model: Any, features: list[str], linear_selector: dict[str, Any] | None):
    def pick(rows: list[dict[str, Any]]) -> dict[str, Any]:
        matrix = np.asarray([vector(row, features, linear_selector) for row in rows], dtype=np.float32)
        predictions = model.predict(matrix)
        best_idx = int(np.argmax(predictions))
        rows[best_idx]["nonlinear_selector_score"] = float(predictions[best_idx])
        return rows[best_idx]

    return pick


def make_model(name: str, seed: int):
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=4,
            random_state=seed,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.03,
            max_leaf_nodes=15,
            min_samples_leaf=8,
            l2_regularization=0.05,
            random_state=seed,
        )
    if name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                alpha=0.01,
                learning_rate_init=0.001,
                max_iter=600,
                early_stopping=True,
                random_state=seed,
            ),
        )
    raise ValueError(f"unsupported model: {name}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit nonlinear T1 final selector models locally")
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--eval", action="append", required=True, help="name=path/to/selector_rows.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--features", default=",".join(DEFAULT_NONLINEAR_FEATURES))
    parser.add_argument("--linear-selector", default="")
    parser.add_argument(
        "--models",
        default="extra_trees,random_forest,gradient_boosting,hist_gradient_boosting,mlp",
    )
    parser.add_argument("--target", default="neg_regret", choices=["neg_regret", "score", "zero_one"])
    parser.add_argument("--seed", type=int, default=20260603)
    args = parser.parse_args(list(argv) if argv is not None else None)

    features = [part.strip() for part in args.features.split(",") if part.strip()]
    train_specs = [parse_spec(raw) for raw in args.train]
    eval_specs = [parse_spec(raw) for raw in args.eval]
    train_groups = group_rows(train_specs)
    if not train_groups:
        raise SystemExit("no T1 refined train groups found")
    linear_selector = load_json(args.linear_selector)
    x_train, y_train, weights = build_matrix(
        train_groups,
        features,
        target=args.target,
        linear_selector=linear_selector,
    )

    eval_groups_by_name = {spec.name: group_rows([spec]) for spec in eval_specs}
    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train": {spec.name: str(spec.path) for spec in train_specs},
        "eval": {spec.name: str(spec.path) for spec in eval_specs},
        "features": features,
        "linear_selector": args.linear_selector,
        "target": args.target,
        "seed": int(args.seed),
        "train_groups": len(train_groups),
        "train_rows": int(x_train.shape[0]),
        "baselines": {},
        "models": {},
    }
    for name, groups in eval_groups_by_name.items():
        report["baselines"][name] = {
            "runtime_best": summarize_choice(groups, pick_runtime),
            "refined_score": summarize_choice(groups, pick_refined),
            "model_score": summarize_choice(groups, pick_model),
        }

    for model_name in [part.strip() for part in args.models.split(",") if part.strip()]:
        model = make_model(model_name, int(args.seed))
        try:
            model.fit(x_train, y_train, sample_weight=weights)
        except (TypeError, ValueError):
            model.fit(x_train, y_train)
        model_report: dict[str, Any] = {}
        for name, groups in eval_groups_by_name.items():
            model_report[name] = summarize_choice(groups, model_picker(model, features, linear_selector))
        report["models"][model_name] = model_report

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "train_groups": len(train_groups)}, indent=2))


if __name__ == "__main__":
    main()
