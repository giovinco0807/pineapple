"""Evaluate T2 candidate union pools from sklearn/LightGBM selectors.

This measures the practical path where several fast selectors propose a
shortlist, then exact rerank chooses the best action inside the union.  A hit
means the teacher-best candidate is included in that union pool.
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
from ai.training.train_t2_sklearn_meta_ranker import load_runtime_features


INACTIVE_SELECTOR_SCORE = -1.0e7


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    target: str


@dataclass(frozen=True)
class SelectorSpec:
    name: str
    model_names: tuple[str, ...]
    gamma: float


@dataclass(frozen=True)
class EvalSpec:
    name: str
    path: Path


@dataclass
class Dataset:
    spec: EvalSpec
    scores: np.ndarray
    base_scores: np.ndarray
    bounds: list[tuple[int, int]]
    selector_scores: dict[str, np.ndarray]


def parse_named_path(value: str) -> EvalSpec:
    parts = value.split("=", 1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise ValueError("expected name=path")
    return EvalSpec(parts[0].strip(), Path(parts[1].strip()))


def parse_model(value: str) -> ModelSpec:
    parts = value.split("=", 1)
    if len(parts) != 2:
        raise ValueError("--model must be name=path,target")
    name = parts[0].strip()
    items = [part.strip() for part in parts[1].split(",")]
    if len(items) != 2:
        raise ValueError("--model must be name=path,target")
    target = items[1].lower()
    if target not in {"score", "residual"}:
        raise ValueError("model target must be score or residual")
    return ModelSpec(name=name, path=Path(items[0]), target=target)


def parse_selector(value: str) -> SelectorSpec:
    parts = value.split("=", 1)
    if len(parts) != 2:
        raise ValueError("--selector must be name=model_a+model_b,gamma")
    name = parts[0].strip()
    items = [part.strip() for part in parts[1].split(",")]
    if len(items) != 2:
        raise ValueError("--selector must be name=model_a+model_b,gamma")
    model_names = tuple(part.strip() for part in items[0].split("+") if part.strip())
    if not name or not model_names:
        raise ValueError("--selector needs a name and at least one model")
    return SelectorSpec(name=name, model_names=model_names, gamma=float(items[1]))


def parse_lgbm_selector(value: str) -> tuple[str, Path]:
    parts = value.split("=", 1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise ValueError("--lgbm-selector must be name=path")
    return parts[0].strip(), Path(parts[1].strip())


def load_selector_scores(
    eval_spec: EvalSpec,
    model_specs: dict[str, ModelSpec],
    selectors: list[SelectorSpec],
    lgbm_selectors: list[tuple[str, Path]],
    include_base: bool,
) -> Dataset:
    x, scores, base_scores, bounds = load_runtime_features(eval_spec.path)
    residuals: dict[str, np.ndarray] = {}
    for name, spec in model_specs.items():
        model = joblib.load(spec.path)
        pred = model.predict(x).astype(np.float32)
        residuals[name] = pred if spec.target == "residual" else pred - base_scores

    selector_scores: dict[str, np.ndarray] = {}
    if include_base:
        selector_scores["base_scores"] = base_scores.astype(np.float32)
    for selector in selectors:
        missing = [name for name in selector.model_names if name not in residuals]
        if missing:
            raise ValueError(f"selector {selector.name} references unknown model(s): {missing}")
        avg_residual = np.mean([residuals[name] for name in selector.model_names], axis=0).astype(np.float32)
        selector_scores[selector.name] = (base_scores + selector.gamma * avg_residual).astype(np.float32)
    for name, path in lgbm_selectors:
        model = joblib.load(path)
        selector_scores[name] = model.predict(x).astype(np.float32)
    return Dataset(
        spec=eval_spec,
        scores=scores,
        base_scores=base_scores,
        bounds=bounds,
        selector_scores=selector_scores,
    )


def evaluate_dataset(
    ds: Dataset,
    pool_ks: list[int],
    thresholds: list[float],
) -> tuple[dict, list[dict]]:
    hits = {k: 0 for k in pool_ks}
    regrets = {k: 0.0 for k in pool_ks}
    sizes = {k: [] for k in pool_ks}
    losses = {k: [] for k in pool_ks}
    rows: list[dict] = []
    selector_names = list(ds.selector_scores)
    for group_id, (start, end) in enumerate(ds.bounds):
        idx = np.arange(start, end, dtype=np.int64)
        true = np.asarray(ds.scores[start:end], dtype=np.float64)
        best_local = int(np.argmax(true))
        best_global = int(idx[best_local])
        best_ev = float(true[best_local])
        orders = {
            name: np.argsort(-np.asarray(scores[start:end], dtype=np.float64))
            for name, scores in ds.selector_scores.items()
        }
        active_masks = {
            name: np.asarray(ds.selector_scores[name][start:end], dtype=np.float64) > INACTIVE_SELECTOR_SCORE
            for name in selector_names
        }
        row = {
            "dataset": ds.spec.name,
            "group_id": int(group_id),
            "size": int(end - start),
            "teacher_best_index": best_global,
            "teacher_best_score": best_ev,
            "pool_metrics": {},
        }
        for k in pool_ks:
            members: set[int] = set()
            by_selector: dict[str, list[int]] = {}
            for name in selector_names:
                local_members = [
                    int(idx[i])
                    for i in orders[name]
                    if bool(active_masks[name][int(i)])
                ][: min(k, len(idx))]
                by_selector[name] = local_members
                members.update(local_members)
            best_in_pool = max((float(ds.scores[i]) for i in members), default=float("-inf"))
            regret = best_ev - best_in_pool
            if best_global in members:
                hits[k] += 1
            regrets[k] += regret
            sizes[k].append(len(members))
            losses[k].append(regret)
            row["pool_metrics"][str(k)] = {
                "pool_size": int(len(members)),
                "hit": bool(best_global in members),
                "ev_loss": float(regret),
                "pool_best_ev": float(best_in_pool),
                "indices": sorted(int(i) for i in members),
                "by_selector": by_selector,
            }
        rows.append(row)

    n = max(len(ds.bounds), 1)
    metrics = {
        "groups": int(len(ds.bounds)),
        "samples": int(len(ds.scores)),
        "selectors": selector_names,
    }
    for k in pool_ks:
        size_arr = np.asarray(sizes[k], dtype=np.float64)
        loss_arr = np.asarray(losses[k], dtype=np.float64)
        metrics[f"union_top{k}_recall"] = hits[k] / n
        metrics[f"union_top{k}_rerank_regret"] = regrets[k] / n
        metrics[f"union_top{k}_avg_pool_size"] = float(size_arr.mean()) if len(size_arr) else 0.0
        metrics[f"union_top{k}_max_pool_size"] = int(size_arr.max()) if len(size_arr) else 0
        metrics[f"union_top{k}_ev_loss_mean"] = float(loss_arr.mean()) if len(loss_arr) else 0.0
        metrics[f"union_top{k}_ev_loss_p95"] = float(np.percentile(loss_arr, 95)) if len(loss_arr) else 0.0
        metrics[f"union_top{k}_ev_loss_p99"] = float(np.percentile(loss_arr, 99)) if len(loss_arr) else 0.0
        metrics[f"union_top{k}_ev_loss_max"] = float(loss_arr.max()) if len(loss_arr) else 0.0
        for threshold in thresholds:
            key = str(threshold).replace(".", "p")
            metrics[f"union_top{k}_ev_loss_gt_{key}"] = int((loss_arr > threshold).sum()) if len(loss_arr) else 0
    return metrics, rows


def aggregate_metrics(dataset_metrics: dict[str, dict], pool_ks: list[int], thresholds: list[float]) -> dict:
    groups = sum(int(m["groups"]) for m in dataset_metrics.values())
    samples = sum(int(m["samples"]) for m in dataset_metrics.values())
    out = {"groups": int(groups), "samples": int(samples)}
    for k in pool_ks:
        hit = 0.0
        regret = 0.0
        size_total = 0.0
        max_size = 0
        loss_values: list[float] = []
        threshold_counts = {threshold: 0 for threshold in thresholds}
        for m in dataset_metrics.values():
            n = int(m["groups"])
            hit += float(m[f"union_top{k}_recall"]) * n
            regret += float(m[f"union_top{k}_rerank_regret"]) * n
            size_total += float(m[f"union_top{k}_avg_pool_size"]) * n
            max_size = max(max_size, int(m[f"union_top{k}_max_pool_size"]))
            # Only p/max summaries are available per dataset.  Keep aggregate
            # mean exact and p/max conservative via max of dataset p/max.
            loss_values.append(float(m[f"union_top{k}_ev_loss_p95"]))
            loss_values.append(float(m[f"union_top{k}_ev_loss_p99"]))
            loss_values.append(float(m[f"union_top{k}_ev_loss_max"]))
            for threshold in thresholds:
                key = str(threshold).replace(".", "p")
                threshold_counts[threshold] += int(m[f"union_top{k}_ev_loss_gt_{key}"])
        denom = max(groups, 1)
        out[f"union_top{k}_recall"] = hit / denom
        out[f"union_top{k}_rerank_regret"] = regret / denom
        out[f"union_top{k}_avg_pool_size"] = size_total / denom
        out[f"union_top{k}_max_pool_size"] = max_size
        out[f"union_top{k}_ev_loss_mean"] = regret / denom
        out[f"union_top{k}_ev_loss_p95_proxy"] = max(loss_values) if loss_values else 0.0
        out[f"union_top{k}_ev_loss_max"] = max(loss_values) if loss_values else 0.0
        for threshold, count in threshold_counts.items():
            key = str(threshold).replace(".", "p")
            out[f"union_top{k}_ev_loss_gt_{key}"] = int(count)
    return out


def write_markdown(path: Path, summary: dict, pool_ks: list[int]) -> None:
    lines = [
        "# T2 Sklearn Candidate Union",
        "",
        f"- eval: `{', '.join(summary['datasets'].keys())}`",
        f"- selectors: `{', '.join(summary['selectors'])}`",
        f"- groups: `{summary['aggregate']['groups']}`",
        "",
        "| per-selector K | recall | mean EV loss | max EV loss | avg pool | max pool |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    agg = summary["aggregate"]
    for k in pool_ks:
        lines.append(
            f"| {k} | "
            f"{float(agg[f'union_top{k}_recall']):.1%} | "
            f"{float(agg[f'union_top{k}_ev_loss_mean']):.3f} | "
            f"{float(agg[f'union_top{k}_ev_loss_max']):.3f} | "
            f"{float(agg[f'union_top{k}_avg_pool_size']):.1f} | "
            f"{int(agg[f'union_top{k}_max_pool_size'])} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    started = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_specs = [parse_named_path(value) for value in args.eval_data]
    model_specs = {spec.name: spec for spec in [parse_model(value) for value in args.model]}
    selectors = [parse_selector(value) for value in args.selector]
    lgbm_selectors = [parse_lgbm_selector(value) for value in args.lgbm_selector]
    pool_ks = parse_topks(args.pool_ks)
    thresholds = [float(part) for part in args.ev_loss_thresholds.split(",") if part.strip()]

    datasets: dict[str, dict] = {}
    all_rows: list[dict] = []
    selector_names: list[str] = []
    for spec in eval_specs:
        ds = load_selector_scores(
            spec,
            model_specs,
            selectors,
            lgbm_selectors,
            include_base=not args.no_base,
        )
        if not selector_names:
            selector_names = list(ds.selector_scores)
        metrics, rows = evaluate_dataset(ds, pool_ks, thresholds)
        datasets[spec.name] = {"path": str(spec.path), "metrics": metrics}
        all_rows.extend(rows)
    dataset_metrics = {name: item["metrics"] for name, item in datasets.items()}
    summary = {
        "models": {name: {"path": str(spec.path), "target": spec.target} for name, spec in model_specs.items()},
        "selectors": selector_names,
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
    parser.add_argument("--no-base", action="store_true")
    parser.add_argument("--pool-ks", default="1,2,3,5,8,10,15,20")
    parser.add_argument("--ev-loss-thresholds", default="0.05,0.1,0.25,0.5,1.0")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selector and not args.lgbm_selector and args.no_base:
        raise ValueError("At least one selector is required")
    run(args)


if __name__ == "__main__":
    main()
