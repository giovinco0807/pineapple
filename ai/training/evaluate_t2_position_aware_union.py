"""Evaluate position-aware T2 Top1 and union-pool exact-rerank safety.

This diagnostic reuses the selector-feature stack from a saved
``train_t2_selector_feature_ranker`` run.  It does not train anything.  It
loads external exact/capped-eval datasets, scores candidates with saved
sklearn models, then reports:

* single-model TopK metrics
* position-aware Top1 policies such as BB=HGB classifier and BTN=ExtraTrees
* exact-rerank safety for small union pools built from several model TopK lists
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np

from ai.training.evaluate_action_value_reranker import parse_topks
from ai.training.evaluate_action_value_set_reranker import summarize_groups
from ai.training.train_t2_selector_feature_ranker import (
    DataSpec,
    ModelSpec,
    SelectorData,
    load_selector_scores,
    model_features,
    parse_lgbm_selector,
    parse_model,
    parse_named_path,
    parse_selector,
)


def repeated_flag_values(command: list[str], flag: str) -> list[str]:
    out: list[str] = []
    for idx, item in enumerate(command[:-1]):
        if item == flag:
            out.append(command[idx + 1])
    return out


def load_stack_specs(training_run: Path) -> tuple[list[ModelSpec], list[Any], list[tuple[str, Path]], list[str]]:
    command_path = training_run / "train_command.json"
    with command_path.open("r", encoding="utf-8") as f:
        command = json.load(f)
    model_specs = [parse_model(value) for value in repeated_flag_values(command, "--model")]
    selector_specs = [parse_selector(value) for value in repeated_flag_values(command, "--selector")]
    lgbm_selectors = [parse_lgbm_selector(value) for value in repeated_flag_values(command, "--lgbm-selector")]
    npy_selectors = repeated_flag_values(command, "--npy-selector")
    return model_specs, selector_specs, lgbm_selectors, npy_selectors


def parse_named_path_value(value: str) -> tuple[str, Path]:
    name, sep, path = value.partition("=")
    if not sep or not name.strip() or not path.strip():
        raise ValueError(f"expected name=path, got {value!r}")
    return name.strip(), Path(path.strip())


def classifier_or_regressor_scores(model: object, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return np.asarray(proba[:, 1], dtype=np.float32)
    return np.asarray(model.predict(x), dtype=np.float32)


def load_positions(data_dir: Path, n_samples: int) -> np.ndarray:
    path = data_dir / "positions.npy"
    if not path.exists():
        return np.asarray(["unknown"] * n_samples, dtype=object)
    raw = np.load(path, mmap_mode="r")[:n_samples]
    positions: list[str] = []
    for value in raw:
        if isinstance(value, bytes):
            text = value.decode("utf-8")
        else:
            text = str(value)
        if text == "0":
            text = "bb"
        elif text == "1":
            text = "btn"
        positions.append(text.lower())
    return np.asarray(positions, dtype=object)


def group_positions(positions: np.ndarray, bounds: list[tuple[int, int]]) -> list[str]:
    out: list[str] = []
    for start, end in bounds:
        values = [str(v).lower() for v in positions[start:end]]
        if not values:
            out.append("unknown")
            continue
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        out.append(max(counts.items(), key=lambda item: item[1])[0])
    return out


def dataset_predictions(
    data: SelectorData,
    final_models: dict[str, object],
) -> dict[str, np.ndarray]:
    x = model_features([data], include_state=True)
    return {name: classifier_or_regressor_scores(model, x) for name, model in final_models.items()}


def position_aware_prediction(
    data: SelectorData,
    predictions: dict[str, np.ndarray],
    *,
    bb_model: str,
    btn_model: str,
    fallback_model: str,
) -> np.ndarray:
    positions = group_positions(load_positions(data.spec.path, len(data.scores)), data.bounds)
    out = np.empty(len(data.scores), dtype=np.float32)
    for group_pos, (start, end) in zip(positions, data.bounds):
        if group_pos == "bb":
            model_name = bb_model
        elif group_pos == "btn":
            model_name = btn_model
        else:
            model_name = fallback_model
        out[start:end] = predictions[model_name][start:end]
    return out


def summarize_prediction(data: SelectorData, pred: np.ndarray, topks: list[int]) -> tuple[dict[str, Any], list[dict]]:
    metrics, rows = summarize_groups(data.scores, pred.astype(np.float32), data.bounds, topks)
    compact = {
        "groups": metrics.get("groups", 0),
        "samples": metrics.get("samples", 0),
        "group_top1_regret": metrics.get("group_top1_regret", 0.0),
    }
    for key in topks:
        compact[f"group_top{key}"] = metrics.get(f"group_top{key}", 0.0)
        compact[f"group_top{key}_rerank_regret"] = metrics.get(f"group_top{key}_rerank_regret", 0.0)
    return compact, rows


def top_local_indices(pred: np.ndarray, start: int, end: int, k: int) -> list[int]:
    kk = min(max(int(k), 0), end - start)
    if kk <= 0:
        return []
    ordered = np.argsort(-np.asarray(pred[start:end], dtype=np.float64))[:kk]
    return [int(i) for i in ordered]


def summarize_union_pool(
    data: SelectorData,
    predictions: dict[str, np.ndarray],
    specs: list[tuple[str, int]],
) -> dict[str, Any]:
    positions = group_positions(load_positions(data.spec.path, len(data.scores)), data.bounds)
    hits = 0
    regret_sum = 0.0
    max_regret = 0.0
    pool_sizes: list[int] = []
    miss_rows: list[dict[str, Any]] = []
    for group_id, (start, end) in enumerate(data.bounds):
        true = np.asarray(data.scores[start:end], dtype=np.float64)
        if len(true) == 0:
            continue
        true_best = int(np.argmax(true))
        true_best_score = float(true[true_best])
        pool: set[int] = set()
        for model_name, k in specs:
            if model_name == "position_aware_cls":
                model = "hgb_cls_l31_state" if positions[group_id] == "bb" else "extra_trees_cls_d12_state"
                pool.update(top_local_indices(predictions[model], start, end, k))
            else:
                pool.update(top_local_indices(predictions[model_name], start, end, k))
        if not pool:
            continue
        pool_list = sorted(pool)
        pool_best = max(float(true[idx]) for idx in pool_list)
        regret = max(0.0, true_best_score - pool_best)
        if regret <= 1e-9:
            hits += 1
        else:
            miss_rows.append(
                {
                    "group_id": int(group_id),
                    "position": positions[group_id],
                    "best_local": int(true_best),
                    "best_ev": true_best_score,
                    "pool_size": int(len(pool_list)),
                    "ev_loss": float(regret),
                    "pool_locals": [int(i) for i in pool_list],
                }
            )
        regret_sum += regret
        max_regret = max(max_regret, regret)
        pool_sizes.append(len(pool_list))
    n_groups = max(len(pool_sizes), 1)
    return {
        "groups": int(len(pool_sizes)),
        "hit": hits / n_groups,
        "regret": regret_sum / n_groups,
        "max_regret": max_regret,
        "avg_pool": float(np.mean(pool_sizes)) if pool_sizes else 0.0,
        "p95_pool": float(np.percentile(np.asarray(pool_sizes), 95)) if pool_sizes else 0.0,
        "max_pool": int(max(pool_sizes)) if pool_sizes else 0,
        "miss_count": int(len(miss_rows)),
        "miss_rows": miss_rows,
    }


def parse_union(value: str) -> tuple[str, list[tuple[str, int]]]:
    name, sep, raw_specs = value.partition("=")
    if not sep or not name.strip() or not raw_specs.strip():
        raise ValueError("--union must be name=model:k,model:k")
    specs: list[tuple[str, int]] = []
    for part in raw_specs.split(","):
        model_name, sep2, raw_k = part.partition(":")
        if not sep2:
            raise ValueError(f"bad union spec part {part!r}")
        specs.append((model_name.strip(), int(raw_k)))
    return name.strip(), specs


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# T2 Position-Aware / Union Evaluation",
        "",
        f"- training_run: `{summary['training_run']}`",
        f"- elapsed_seconds: `{summary['elapsed_seconds']:.3f}`",
        "",
    ]
    for ds_name, result in summary["datasets"].items():
        lines.extend(
            [
                f"## {ds_name}",
                "",
                f"- data: `{result['path']}`",
                f"- groups: `{result['groups']}`",
                f"- positions: `{result['positions']}`",
                "",
                "### Single / Position-Aware",
                "",
                "| selector | Top1 | Reg1 | Top3 | Top5 | Top8 | Top10 | Top20 | Reg10 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in result["single_rows"]:
            m = row["metrics"]
            lines.append(
                f"| {row['name']} | "
                f"{float(m.get('group_top1', 0.0)):.1%} | "
                f"{float(m.get('group_top1_regret', 0.0)):.3f} | "
                f"{float(m.get('group_top3', 0.0)):.1%} | "
                f"{float(m.get('group_top5', 0.0)):.1%} | "
                f"{float(m.get('group_top8', 0.0)):.1%} | "
                f"{float(m.get('group_top10', 0.0)):.1%} | "
                f"{float(m.get('group_top20', 0.0)):.1%} | "
                f"{float(m.get('group_top10_rerank_regret', 0.0)):.3f} |"
            )
        lines.extend(
            [
                "",
                "### Union Exact-Rerank Pool",
                "",
                "| pool | hit | avg EV loss | max EV loss | avg pool | p95 pool | max pool | misses |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in result["union_rows"]:
            m = row["metrics"]
            lines.append(
                f"| {row['name']} | "
                f"{float(m['hit']):.1%} | "
                f"{float(m['regret']):.3f} | "
                f"{float(m['max_regret']):.3f} | "
                f"{float(m['avg_pool']):.1f} | "
                f"{float(m['p95_pool']):.1f} | "
                f"{int(m['max_pool'])} | "
                f"{int(m['miss_count'])} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    started = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    topks = parse_topks(args.topks)
    training_run = Path(args.training_run)
    model_specs, selector_specs, lgbm_selectors, npy_selectors = load_stack_specs(training_run)
    loaded_base = {spec.name: joblib.load(spec.path) for spec in model_specs}
    loaded_lgbm = {name: joblib.load(path) for name, path in lgbm_selectors}
    final_models = {name: joblib.load(path) for name, path in [parse_named_path_value(v) for v in args.final_model]}
    eval_specs = [parse_named_path(value) for value in args.eval_data]
    union_specs = [parse_union(value) for value in args.union]

    datasets: dict[str, Any] = {}
    for spec in eval_specs:
        data = load_selector_scores(
            spec,
            {s.name: s for s in model_specs},
            selector_specs,
            lgbm_selectors,
            loaded_base,
            loaded_lgbm,
            include_base=True,
            npy_selectors=npy_selectors,
        )
        predictions = dataset_predictions(data, final_models)
        positions = group_positions(load_positions(data.spec.path, len(data.scores)), data.bounds)
        pos_counts: dict[str, int] = {}
        for pos in positions:
            pos_counts[pos] = pos_counts.get(pos, 0) + 1

        named_predictions = dict(predictions)
        named_predictions["posaware_hgb_bb_extra_btn"] = position_aware_prediction(
            data,
            predictions,
            bb_model="hgb_cls_l31_state",
            btn_model="extra_trees_cls_d12_state",
            fallback_model="extra_trees_cls_d12_state",
        )
        named_predictions["posaware_hgb_bb_oldev_btn"] = position_aware_prediction(
            data,
            predictions,
            bb_model="hgb_cls_l31_state",
            btn_model="old_ev",
            fallback_model="old_ev",
        )

        single_rows = []
        for name, pred in named_predictions.items():
            metrics, _rows = summarize_prediction(data, pred, topks)
            single_rows.append({"name": name, "metrics": metrics})
        single_rows.sort(
            key=lambda row: (
                float(row["metrics"].get("group_top1", 0.0)),
                -float(row["metrics"].get("group_top1_regret", 0.0)),
                float(row["metrics"].get("group_top10", 0.0)),
            ),
            reverse=True,
        )

        union_rows = []
        for name, specs in union_specs:
            metrics = summarize_union_pool(data, predictions, specs)
            union_rows.append({"name": name, "specs": specs, "metrics": metrics})
        union_rows.sort(
            key=lambda row: (
                float(row["metrics"]["hit"]),
                -float(row["metrics"]["regret"]),
                -float(row["metrics"]["avg_pool"]),
            ),
            reverse=True,
        )

        datasets[spec.name] = {
            "path": str(spec.path),
            "groups": int(len(data.bounds)),
            "samples": int(len(data.scores)),
            "positions": pos_counts,
            "single_rows": single_rows,
            "union_rows": union_rows,
        }

    summary = {
        "training_run": str(training_run),
        "models": {name: str(path) for name, path in [parse_named_path_value(v) for v in args.final_model]},
        "eval_data": {spec.name: str(spec.path) for spec in eval_specs},
        "topks": topks,
        "unions": {name: specs for name, specs in union_specs},
        "datasets": datasets,
        "elapsed_seconds": time.time() - started,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_markdown(out_dir / "summary.md", summary)
    print(json.dumps({"summary": str(out_dir / "summary.json"), "elapsed_seconds": summary["elapsed_seconds"]}, indent=2))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--eval-data", action="append", required=True, help="name=path")
    parser.add_argument("--final-model", action="append", required=True, help="name=path")
    parser.add_argument("--union", action="append", default=[], help="name=model:k,model:k")
    parser.add_argument("--topks", default="1,3,5,8,10,15,20")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.union:
        args.union = [
            "cls1_each=hgb_cls_l31_state:1,extra_trees_cls_d12_state:1",
            "cls3_each=hgb_cls_l31_state:3,extra_trees_cls_d12_state:3",
            "all6_top1=hgb_cls_l31_state:1,extra_trees_cls_d12_state:1,old_score:1,old_ev:1,latest_score:1,latest_ev:1",
            "all6_top3=hgb_cls_l31_state:3,extra_trees_cls_d12_state:3,old_score:3,old_ev:3,latest_score:3,latest_ev:3",
            "all6_top5=hgb_cls_l31_state:5,extra_trees_cls_d12_state:5,old_score:5,old_ev:5,latest_score:5,latest_ev:5",
            "posaware1_safety3=position_aware_cls:1,old_score:3,old_ev:3,latest_score:3,latest_ev:3",
        ]
    run(args)


if __name__ == "__main__":
    main()
