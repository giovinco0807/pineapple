"""Evaluate selector-feature T2 models with a runtime-only bonus selector.

This diagnostic keeps the trained selector-feature model unchanged, then adds a
small runtime-only bonus from ``selector_scores/<name>.npy`` to candidates where
the selector fires.  Teacher EV is used only for evaluation.
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

from ai.training.train_t2_neural_pool_reranker import SELECTOR_DEFINITIONS
from ai.training.train_t2_selector_feature_ranker import (
    DataSpec,
    ModelSpec,
    SelectorSpec,
    load_selector_scores,
    model_features,
    parse_lgbm_selector,
    parse_named_path,
    parse_selector,
)


INACTIVE_SELECTOR_SCORE = -1.0e7


def parse_fixed_bonuses(values: list[str] | None) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for value in values or []:
        name, sep, weight = value.partition("=")
        if not sep or not name.strip() or not weight.strip():
            raise ValueError("--fixed-bonus must be name=weight")
        out.append((name.strip(), float(weight.strip())))
    return out


def group_bounds(group_ids: np.ndarray) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    if len(group_ids) == 0:
        return bounds
    start = 0
    current = int(group_ids[0])
    for idx, value in enumerate(group_ids[1:], start=1):
        if int(value) != current:
            bounds.append((start, idx))
            start = idx
            current = int(value)
    bounds.append((start, len(group_ids)))
    return bounds


def split_summary_selectors(
    config: dict,
) -> tuple[bool, list[SelectorSpec], list[tuple[str, Path]], list[str]]:
    include_base = False
    selectors: list[SelectorSpec] = []
    lgbm_names = set((config.get("lgbm_selectors") or {}).keys())
    npy_selectors: list[str] = []
    for name in config.get("selectors", []):
        if name == "base_scores":
            include_base = True
        elif name in SELECTOR_DEFINITIONS:
            selectors.append(parse_selector(SELECTOR_DEFINITIONS[name]))
        elif name in lgbm_names:
            continue
        else:
            npy_selectors.append(name)
    lgbm_selectors = [
        parse_lgbm_selector(f"{name}={path}")
        for name, path in (config.get("lgbm_selectors") or {}).items()
    ]
    return include_base, selectors, lgbm_selectors, npy_selectors


def predict_model(model: object, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if getattr(proba, "ndim", 1) == 2 and proba.shape[1] >= 2:
            return np.asarray(proba[:, 1], dtype=np.float32)
    return np.asarray(model.predict(x), dtype=np.float32)


def evaluate_scores(
    scores: np.ndarray,
    pred: np.ndarray,
    bounds: list[tuple[int, int]],
    topks: list[int],
    epsilon: float,
) -> dict:
    topk_hits = {k: 0 for k in topks}
    topk_regrets = {k: [] for k in topks}
    top1_ev_hits = 0
    top1_index_misses = 0
    top1_ev_loss_misses = 0
    top1_ev_tie_misses = 0
    top1_losses: list[float] = []

    for start, end in bounds:
        true = np.asarray(scores[start:end], dtype=np.float64)
        estimate = np.asarray(pred[start:end], dtype=np.float64)
        if len(true) == 0:
            continue
        best_local = int(np.argmax(true))
        order = [int(idx) for idx in np.argsort(-estimate)]
        selected_local = int(order[0])
        best_score = float(true[best_local])
        selected_score = float(true[selected_local])
        top1_loss = max(0.0, best_score - selected_score)
        top1_losses.append(top1_loss)

        if top1_loss <= epsilon:
            top1_ev_hits += 1
        if selected_local != best_local:
            top1_index_misses += 1
            if top1_loss <= epsilon:
                top1_ev_tie_misses += 1
            else:
                top1_ev_loss_misses += 1

        for k in topks:
            shortlist = order[: min(k, len(order))]
            topk_hits[k] += int(best_local in set(shortlist))
            best_in_shortlist = max(float(true[idx]) for idx in shortlist)
            topk_regrets[k].append(max(0.0, best_score - best_in_shortlist))

    groups = max(len(bounds), 1)
    losses = np.asarray(top1_losses, dtype=np.float64)
    out = {
        "groups": int(len(bounds)),
        "samples": int(len(scores)),
        "group_top1_ev_hit": float(top1_ev_hits / groups),
        "group_top1_index_miss_count": int(top1_index_misses),
        "group_top1_ev_loss_miss_count": int(top1_ev_loss_misses),
        "group_top1_ev_tie_miss_count": int(top1_ev_tie_misses),
        "group_top1_ev_loss_mean": float(losses.mean()) if len(losses) else 0.0,
        "group_top1_ev_loss_max": float(losses.max()) if len(losses) else 0.0,
        "group_top1_ev_hit_epsilon": float(epsilon),
    }
    for k in topks:
        regrets = np.asarray(topk_regrets[k], dtype=np.float64)
        out[f"group_top{k}"] = float(topk_hits[k] / groups)
        out[f"group_top{k}_rerank_regret"] = float(regrets.mean()) if len(regrets) else 0.0
        out[f"group_top{k}_rerank_regret_max"] = float(regrets.max()) if len(regrets) else 0.0
    return out


def aggregate_dataset_metrics(rows: list[dict], topks: list[int], epsilon: float) -> dict:
    scores = np.concatenate([row["scores"] for row in rows]) if rows else np.asarray([], dtype=np.float32)
    pred = np.concatenate([row["pred"] for row in rows]) if rows else np.asarray([], dtype=np.float32)
    bounds: list[tuple[int, int]] = []
    offset = 0
    for row in rows:
        for start, end in row["bounds"]:
            bounds.append((start + offset, end + offset))
        offset += len(row["scores"])
    return evaluate_scores(scores, pred, bounds, topks, epsilon)


def load_bonus_values(data_path: Path, name: str, length: int, missing_zero: bool) -> np.ndarray:
    selector_path = data_path / "selector_scores" / f"{name}.npy"
    if selector_path.exists():
        selector = np.asarray(np.load(selector_path), dtype=np.float64)
        if len(selector) != length:
            raise ValueError(f"{name}: bonus selector length {len(selector)} != predictions {length}")
        return selector
    if missing_zero:
        return np.full(length, INACTIVE_SELECTOR_SCORE, dtype=np.float64)
    raise FileNotFoundError(f"missing bonus selector: {selector_path}")


def load_config_datasets(args: argparse.Namespace, config: dict):
    model_specs = {
        name: ModelSpec(name=name, path=Path(item["path"]), target=str(item["target"]))
        for name, item in config.get("models", {}).items()
    }
    include_base, selectors, lgbm_selectors, npy_selectors = split_summary_selectors(config)
    loaded_models = {name: joblib.load(spec.path) for name, spec in model_specs.items()}
    loaded_lgbm = {name: joblib.load(path) for name, path in lgbm_selectors}
    specs = [parse_named_path(value) for value in args.eval_data]
    datasets = [
        load_selector_scores(
            spec,
            model_specs=model_specs,
            selectors=selectors,
            lgbm_selectors=lgbm_selectors,
            loaded_models=loaded_models,
            loaded_lgbm=loaded_lgbm,
            include_base=include_base,
            npy_selectors=npy_selectors,
        )
        for spec in specs
    ]
    return datasets, {
        "include_base": include_base,
        "selectors": [selector.name for selector in selectors],
        "lgbm_selectors": [name for name, _path in lgbm_selectors],
        "npy_selectors": npy_selectors,
    }


def run(args: argparse.Namespace) -> None:
    config_path = Path(args.summary)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = joblib.load(args.model_path)
    datasets, selector_sources = load_config_datasets(args, config)
    topks = [int(part) for part in args.topks.split(",") if part.strip()]
    bonuses = [float(part) for part in args.bonuses.split(",") if part.strip()]
    fixed_bonuses = parse_fixed_bonuses(args.fixed_bonus)

    output = {
        "summary": str(config_path),
        "model_path": str(Path(args.model_path)),
        "include_state": bool(args.include_state),
        "bonus_selector": args.bonus_selector,
        "bonuses": bonuses,
        "fixed_bonuses": [
            {"selector": name, "weight": float(weight)}
            for name, weight in fixed_bonuses
        ],
        "topks": topks,
        "ev_hit_epsilon": float(args.ev_hit_epsilon),
        "selector_sources": selector_sources,
        "datasets": {},
        "aggregate": {},
    }

    for bonus in bonuses:
        aggregate_rows: list[dict] = []
        output["aggregate"][str(bonus)] = {}
        for ds in datasets:
            x = model_features([ds], include_state=args.include_state)
            expected = getattr(model, "n_features_in_", x.shape[1])
            if int(expected) != int(x.shape[1]):
                raise ValueError(
                    f"{ds.spec.name}: model expects {expected} features, built {x.shape[1]}; "
                    "check --include-state and selector summary"
                )
            pred = predict_model(model, x)
            adjusted = np.asarray(pred, dtype=np.float64)
            fixed_active_candidates = 0
            fixed_active_groups: set[int] = set()
            group_ids = np.load(ds.spec.path / "group_ids.npy")
            for fixed_name, fixed_weight in fixed_bonuses:
                fixed_selector = load_bonus_values(
                    ds.spec.path,
                    fixed_name,
                    len(pred),
                    args.missing_bonus_zero,
                )
                fixed_active = fixed_selector > INACTIVE_SELECTOR_SCORE
                adjusted[fixed_active] += fixed_selector[fixed_active] * fixed_weight
                fixed_active_candidates += int(fixed_active.sum())
                if fixed_active.any():
                    fixed_active_groups.update(int(value) for value in group_ids[fixed_active])

            selector = load_bonus_values(
                ds.spec.path,
                args.bonus_selector,
                len(pred),
                args.missing_bonus_zero,
            )
            active = selector > INACTIVE_SELECTOR_SCORE
            adjusted[active] += selector[active] * bonus
            metrics = evaluate_scores(ds.scores, adjusted, ds.bounds, topks, args.ev_hit_epsilon)
            metrics["active_candidates"] = int(active.sum())
            metrics["active_groups"] = int(len(set(group_ids[active])) if active.any() else 0)
            metrics["fixed_active_candidates"] = int(fixed_active_candidates)
            metrics["fixed_active_groups"] = int(len(fixed_active_groups))
            output["datasets"].setdefault(ds.spec.name, {"path": str(ds.spec.path), "bonuses": {}})
            output["datasets"][ds.spec.name]["bonuses"][str(bonus)] = metrics
            aggregate_rows.append({"scores": ds.scores, "pred": adjusted, "bounds": ds.bounds})
        output["aggregate"][str(bonus)] = aggregate_dataset_metrics(
            aggregate_rows, topks, args.ev_hit_epsilon
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(out_path),
                "aggregate": output["aggregate"],
                "datasets": {
                    name: value["bonuses"] for name, value in output["datasets"].items()
                },
            },
            indent=2,
        )
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="selector-feature training summary.json")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--eval-data", action="append", required=True, help="name=path")
    parser.add_argument("--bonus-selector", required=True)
    parser.add_argument("--bonuses", default="0,0.002,0.005,0.01,0.02,0.05")
    parser.add_argument(
        "--fixed-bonus",
        action="append",
        default=[],
        help="Additional runtime bonus applied on every sweep step, formatted name=weight",
    )
    parser.add_argument("--topks", default="1,3,5,10,15,20")
    parser.add_argument("--ev-hit-epsilon", type=float, default=1.0e-9)
    parser.add_argument("--include-state", action="store_true")
    parser.add_argument("--missing-bonus-zero", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    run(args)


if __name__ == "__main__":
    main()
