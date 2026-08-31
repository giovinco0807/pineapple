"""Evaluate T2 union pools including selector-feature rankers.

``evaluate_t2_sklearn_candidate_union.py`` can union base sklearn selectors
whose predictions are made from the runtime state feature vector.  The
selector-feature rankers trained by ``train_t2_selector_feature_ranker.py`` use
an additional derived feature block built from several selector scores/ranks.
This diagnostic script evaluates whether adding those rankers as extra union
sources reduces TopK misses before exact rerank.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.training.evaluate_action_value_reranker import parse_topks
from ai.training.evaluate_t2_sklearn_candidate_union import (
    Dataset,
    EvalSpec,
    aggregate_metrics,
    evaluate_dataset,
    parse_lgbm_selector,
    parse_model,
    parse_named_path,
    parse_selector,
    write_markdown,
)
from ai.training.train_t2_selector_feature_ranker import adapt_runtime_features, load_selector_scores, predict_adapted


@dataclass(frozen=True)
class FeatureSelectorSpec:
    name: str
    path: Path
    scope: str


def parse_feature_selector(value: str) -> FeatureSelectorSpec:
    parts = value.split("=", 1)
    if len(parts) != 2 or not parts[0].strip():
        raise ValueError("--feature-selector must be name=path,selector|state")
    name = parts[0].strip()
    items = [part.strip() for part in parts[1].split(",")]
    if len(items) != 2:
        raise ValueError("--feature-selector must be name=path,selector|state")
    scope = items[1].lower()
    if scope not in {"selector", "state"}:
        raise ValueError("feature selector scope must be selector or state")
    return FeatureSelectorSpec(name=name, path=Path(items[0]), scope=scope)


def feature_model_input(ds, scope: str, model: object | None = None) -> np.ndarray:
    if scope == "selector":
        return ds.selector_features.astype(np.float32)
    expected = getattr(model, "n_features_in_", None)
    if expected is not None:
        expected = int(expected)
        selector_width = int(ds.selector_features.shape[1])
        # Older feature-selector models were trained on the old runtime state
        # prefix plus selector features.  Newer datasets may append board/deck
        # context to ds.x, so keep the selector block instead of blindly taking
        # the first expected columns of the concatenated matrix.
        if expected > selector_width and ds.x.shape[1] + selector_width > expected:
            x_width = expected - selector_width
            if x_width <= ds.x.shape[1]:
                return np.concatenate(
                    [adapt_runtime_features(ds.x, x_width), ds.selector_features],
                    axis=1,
                ).astype(np.float32)
    return np.concatenate([ds.x, ds.selector_features], axis=1).astype(np.float32)


def run(args: argparse.Namespace) -> None:
    started = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_specs = [parse_named_path(value) for value in args.eval_data]
    model_specs = {spec.name: spec for spec in [parse_model(value) for value in args.model]}
    selectors = [parse_selector(value) for value in args.selector]
    lgbm_selectors = [parse_lgbm_selector(value) for value in args.lgbm_selector]
    feature_selectors = [parse_feature_selector(value) for value in args.feature_selector]
    pool_ks = parse_topks(args.pool_ks)
    thresholds = [float(part) for part in args.ev_loss_thresholds.split(",") if part.strip()]

    loaded_models = {name: joblib.load(spec.path) for name, spec in model_specs.items()}
    loaded_lgbm = {name: joblib.load(path) for name, path in lgbm_selectors}
    loaded_feature = {spec.name: joblib.load(spec.path) for spec in feature_selectors}

    datasets: dict[str, dict] = {}
    all_rows: list[dict] = []
    selector_names: list[str] = []
    for spec in eval_specs:
        selector_data = load_selector_scores(
            spec,
            model_specs,
            selectors,
            lgbm_selectors,
            loaded_models,
            loaded_lgbm,
            include_base=not args.no_base,
            npy_selectors=args.npy_selector,
        )
        selector_scores = dict(selector_data.selector_scores)
        for feature_spec in feature_selectors:
            pred = predict_adapted(
                loaded_feature[feature_spec.name],
                feature_model_input(selector_data, feature_spec.scope, loaded_feature[feature_spec.name]),
            )
            selector_scores[feature_spec.name] = np.asarray(pred, dtype=np.float32)
        ds = Dataset(
            spec=EvalSpec(selector_data.spec.name, selector_data.spec.path),
            scores=selector_data.scores,
            base_scores=selector_data.base_scores,
            bounds=selector_data.bounds,
            selector_scores=selector_scores,
        )
        if not selector_names:
            selector_names = list(ds.selector_scores)
        metrics, rows = evaluate_dataset(ds, pool_ks, thresholds)
        datasets[ds.spec.name] = {"path": str(ds.spec.path), "metrics": metrics}
        all_rows.extend(rows)

    dataset_metrics = {name: item["metrics"] for name, item in datasets.items()}
    summary = {
        "models": {name: {"path": str(spec.path), "target": spec.target} for name, spec in model_specs.items()},
        "selectors": selector_names,
        "lgbm_selectors": {name: str(path) for name, path in lgbm_selectors},
        "feature_selectors": {
            spec.name: {"path": str(spec.path), "scope": spec.scope} for spec in feature_selectors
        },
        "datasets": datasets,
        "aggregate": aggregate_metrics(dataset_metrics, pool_ks, thresholds),
        "pool_ks": pool_ks,
        "elapsed_seconds": time.time() - started,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (out_dir / "rows.jsonl").open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    write_markdown(out_dir / "summary.md", summary, pool_ks)
    print(json.dumps({"summary": str(out_dir / "summary.json"), "elapsed_seconds": summary["elapsed_seconds"]}, indent=2))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-data", action="append", required=True, help="name=path")
    parser.add_argument("--model", action="append", default=[], help="name=path,target")
    parser.add_argument("--selector", action="append", default=[], help="name=model_a+model_b,gamma")
    parser.add_argument("--lgbm-selector", action="append", default=[], help="name=path")
    parser.add_argument(
        "--npy-selector",
        action="append",
        default=[],
        help="Selector score stored as <data_dir>/selector_scores/<name>.npy",
    )
    parser.add_argument("--feature-selector", action="append", default=[], help="name=path,selector|state")
    parser.add_argument("--no-base", action="store_true")
    parser.add_argument("--pool-ks", default="1,3,5,8,10,15,20")
    parser.add_argument("--ev-loss-thresholds", default="0.05,0.1,0.25,0.5,1.0")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if (
        not args.selector
        and not args.lgbm_selector
        and not args.npy_selector
        and not args.feature_selector
        and args.no_base
    ):
        raise ValueError("At least one selector source is required")
    run(args)


if __name__ == "__main__":
    main()
