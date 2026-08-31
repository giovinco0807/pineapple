"""Evaluate a T2 pool reranker with an optional runtime-only override selector.

This is for diagnostics where the production candidate scorer remains a trained
pool reranker, but a very narrow runtime rule can override the final Top1 choice
when its selector is active.  Teacher EV is used only for evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.training.evaluate_action_value_set_reranker import summarize_groups
from ai.training.evaluate_t2_selector_feature_union import parse_feature_selector
from ai.training.evaluate_t2_sklearn_candidate_union import parse_lgbm_selector, parse_selector
from ai.training.train_t2_neural_pool_reranker import SELECTOR_DEFINITIONS
from ai.training.train_t2_selector_feature_pool_reranker import load_dataset, predict_dataset
from ai.training.train_t2_selector_feature_ranker import parse_model, parse_named_path


INACTIVE_SELECTOR_SCORE = -1.0e7


def compact(metrics: dict) -> dict:
    return {
        key: value
        for key, value in metrics.items()
        if key in {"groups", "samples"} or key.startswith("group_top") or key.endswith("regret")
    }


def load_eval_sets(args: argparse.Namespace, config: dict) -> list:
    model_specs = {
        spec.name: spec
        for spec in [
            parse_model(f"{name}={item['path']},{item['target']}")
            for name, item in config["models"].items()
        ]
    }
    selector_names = args.selector or [
        name
        for name in config.get("selectors", [])
        if name in SELECTOR_DEFINITIONS
    ]
    selectors = [parse_selector(SELECTOR_DEFINITIONS[name]) for name in selector_names]
    lgbm_selectors = [
        parse_lgbm_selector(f"{name}={path}")
        for name, path in config.get("lgbm_selectors", {}).items()
    ]
    feature_selectors = [
        parse_feature_selector(f"{name}={item['path']},{item['scope']}")
        for name, item in config.get("feature_selectors", {}).items()
    ]
    loaded_models = {name: joblib.load(spec.path) for name, spec in model_specs.items()}
    loaded_lgbm = {name: joblib.load(path) for name, path in lgbm_selectors}
    loaded_feature = {spec.name: joblib.load(spec.path) for spec in feature_selectors}
    specs = [parse_named_path(value) for value in args.eval_data]
    return [
        load_dataset(
            spec,
            model_specs,
            selectors,
            lgbm_selectors,
            args.npy_selector,
            loaded_models,
            loaded_lgbm,
            feature_selectors,
            loaded_feature,
            include_base=not args.no_base,
        )
        for spec in specs
    ]


def apply_override(ds, pred: np.ndarray, selector_name: str) -> tuple[np.ndarray, list[dict]]:
    selector_path = Path(ds.spec.path) / "selector_scores" / f"{selector_name}.npy"
    if not selector_path.exists():
        return pred, []
    selector = np.load(selector_path)
    out = np.array(pred, copy=True)
    rows: list[dict] = []
    for group_id, (start, end) in enumerate(ds.bounds):
        local = np.asarray(selector[start:end], dtype=np.float64)
        if len(local) == 0 or float(np.max(local)) <= INACTIVE_SELECTOR_SCORE:
            continue
        selected = start + int(np.argmax(local))
        best = start + int(np.argmax(ds.scores[start:end]))
        ev_loss = max(0.0, float(ds.scores[best] - ds.scores[selected]))
        out[start:end] = -1.0e9
        out[selected] = 1.0e9
        rows.append(
            {
                "dataset": ds.spec.name,
                "group_id": int(group_id),
                "selected_index": int(selected),
                "teacher_best_index": int(best),
                "ev_loss": ev_loss,
                "hit_index": bool(selected == best),
                "selector_name": selector_name,
            }
        )
    return out, rows


def top1_ev_equiv_metrics(
    scores: np.ndarray,
    pred_score: np.ndarray,
    bounds: list[tuple[int, int]],
    epsilon: float,
) -> dict:
    groups = 0
    ev_hits = 0
    index_misses = 0
    ev_loss_misses = 0
    ev_tie_misses = 0
    max_loss = 0.0
    for start, end in bounds:
        idx = np.arange(start, end, dtype=np.int64)
        true = np.asarray(scores[idx], dtype=np.float64)
        pred = np.asarray(pred_score[idx], dtype=np.float64)
        if len(idx) == 0 or bool(np.isnan(pred).any()):
            continue
        groups += 1
        true_best_local = int(np.argmax(true))
        pred_best_local = int(np.argmax(pred))
        loss = max(0.0, float(true[true_best_local] - true[pred_best_local]))
        max_loss = max(max_loss, loss)
        index_hit = pred_best_local == true_best_local
        ev_hit = loss <= epsilon
        if ev_hit:
            ev_hits += 1
        if not index_hit:
            index_misses += 1
            if ev_hit:
                ev_tie_misses += 1
            else:
                ev_loss_misses += 1
    denom = max(groups, 1)
    return {
        "group_top1_ev_hit": ev_hits / denom,
        "group_top1_index_miss_count": int(index_misses),
        "group_top1_ev_loss_miss_count": int(ev_loss_misses),
        "group_top1_ev_tie_miss_count": int(ev_tie_misses),
        "group_top1_ev_loss_max": float(max_loss),
        "group_top1_ev_hit_epsilon": float(epsilon),
    }


def evaluate(
    datasets: list,
    pred_by_dataset: dict[str, np.ndarray],
    topks: list[int],
    ev_hit_epsilon: float,
) -> tuple[dict, dict]:
    all_scores: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_bounds: list[tuple[int, int]] = []
    per_dataset: dict[str, dict] = {}
    offset = 0
    for ds in datasets:
        pred = pred_by_dataset[ds.spec.name]
        metrics, _rows = summarize_groups(ds.scores, pred, ds.bounds, topks)
        metrics.update(top1_ev_equiv_metrics(ds.scores, pred, ds.bounds, ev_hit_epsilon))
        per_dataset[ds.spec.name] = compact(metrics)
        all_scores.append(ds.scores)
        all_pred.append(pred)
        for start, end in ds.bounds:
            all_bounds.append((start + offset, end + offset))
        offset += len(ds.scores)
    aggregate, _rows = summarize_groups(
        np.concatenate(all_scores), np.concatenate(all_pred), all_bounds, topks
    )
    aggregate.update(
        top1_ev_equiv_metrics(
            np.concatenate(all_scores),
            np.concatenate(all_pred),
            all_bounds,
            ev_hit_epsilon,
        )
    )
    return compact(aggregate), per_dataset


def run(args: argparse.Namespace) -> None:
    config = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    datasets = load_eval_sets(args, config)
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        matching_rows = [
            row for row in config.get("eval_rows", []) if row.get("model") == args.model_kind
        ]
        if matching_rows:
            model_path = Path(matching_rows[0]["path"])
        elif config.get("eval_rows"):
            model_path = Path(config["eval_rows"][0]["path"])
        else:
            raise ValueError("No --model-path provided and config has no eval_rows")
    model = joblib.load(model_path)

    base_pred = {
        ds.spec.name: predict_dataset(
            ds,
            model,
            args.model_kind,
            int(config["pool_k"]),
            config["feature_scope"],
            bool(config["pool_local_features"]),
        )
        for ds in datasets
    }
    base_aggregate, base_per_dataset = evaluate(
        datasets, base_pred, args.topks, args.ev_hit_epsilon
    )

    override_selector_names = []
    for value in args.override_selector or ["t2_strict_weak_top_guard_tactical"]:
        override_selector_names.extend(name.strip() for name in str(value).split(",") if name.strip())

    override_rows: list[dict] = []
    override_pred = {}
    for ds in datasets:
        pred = base_pred[ds.spec.name]
        rows: list[dict] = []
        for selector_name in override_selector_names:
            pred, selector_rows = apply_override(ds, pred, selector_name)
            rows.extend(selector_rows)
        override_pred[ds.spec.name] = pred
        override_rows.extend(rows)
    override_aggregate, override_per_dataset = evaluate(
        datasets, override_pred, args.topks, args.ev_hit_epsilon
    )

    active_losses = [float(row["ev_loss"]) for row in override_rows]
    summary = {
        "source_summary": str(Path(args.summary)),
        "model_path": str(model_path),
        "model_kind": args.model_kind,
        "eval_data": args.eval_data,
        "npy_selectors": args.npy_selector,
        "override_selector": override_selector_names,
        "base": {"aggregate": base_aggregate, "datasets": base_per_dataset},
        "override": {"aggregate": override_aggregate, "datasets": override_per_dataset},
        "override_rows": override_rows,
        "override_active_groups": int(len(override_rows)),
        "override_bad_groups": int(sum(1 for row in override_rows if row["ev_loss"] > 0.0)),
        "override_ev_loss_mean": float(np.mean(active_losses)) if active_losses else 0.0,
        "override_ev_loss_max": float(np.max(active_losses)) if active_losses else 0.0,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "override": summary["override"]["aggregate"]}, indent=2))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--eval-data", action="append", required=True, help="name=path")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--model-kind", default="hgb_cls_l63")
    parser.add_argument("--selector", action="append", default=[])
    parser.add_argument("--npy-selector", action="append", default=["t2_top_aa_pairdraw_tactical"])
    parser.add_argument(
        "--override-selector",
        action="append",
        default=None,
        help="Runtime override selector name. May be repeated or comma-separated.",
    )
    parser.add_argument("--topks", type=int, nargs="*", default=[1, 3, 5, 10, 15, 20])
    parser.add_argument("--ev-hit-epsilon", type=float, default=1.0e-9)
    parser.add_argument("--no-base", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    run(args)


if __name__ == "__main__":
    main()
