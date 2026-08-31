"""Evaluate T2 set-model blends without teacher-leak features.

This script intentionally uses only runtime-available candidate scores:

- base action-value ensemble scores
- old 520-dim set-model scores
- new 617-dim set-model scores

It does not read candidate_ranks.npy or route_tags.npy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.models.action_value_set_reranker import ActionValueSetReranker
from ai.training.evaluate_action_value_reranker import parse_topks
from ai.training.evaluate_action_value_set_reranker import predict_groups, summarize_groups
from ai.training.train_action_value_set_reranker import (
    CandidateGroupDataset,
    build_group_bounds,
    collate_groups,
)


def parse_dataset(value: str) -> tuple[str, Path, Path]:
    parts = value.split("=", 1)
    if len(parts) != 2:
        raise ValueError("--dataset must be name=dim520,dim617")
    name = parts[0].strip()
    paths = [Path(part.strip()) for part in parts[1].split(",")]
    if len(paths) != 2:
        raise ValueError("--dataset must provide exactly two paths: name=dim520,dim617")
    return name, paths[0], paths[1]


def parse_blend(value: str) -> tuple[str, np.ndarray]:
    parts = value.split("=", 1)
    if len(parts) != 2:
        raise ValueError("--blend must be name=w_old1,w_old2,w_new1,w_new2")
    name = parts[0].strip()
    weights = np.asarray([float(part) for part in parts[1].split(",")], dtype=np.float32)
    if len(weights) != 4:
        raise ValueError("--blend requires four weights")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("--blend weights must sum positive")
    return name, weights / total


def load_arrays(data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], np.ndarray]:
    scores = np.asarray(np.load(data_dir / "scores.npy", mmap_mode="r"), dtype=np.float32)
    group_ids = np.asarray(np.load(data_dir / "group_ids.npy", mmap_mode="r"), dtype=np.int64)
    bounds = build_group_bounds(group_ids)
    base_scores = np.asarray(np.load(data_dir / "base_scores.npy", mmap_mode="r"), dtype=np.float32)
    return scores, group_ids, bounds, base_scores


def checkpoint_cache_key(checkpoint: Path) -> str:
    path = checkpoint.resolve()
    try:
        stat = path.stat()
        raw = f"{path}|{stat.st_size}|{stat.st_mtime_ns}"
    except OSError:
        raw = str(path)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def prediction_cache_path(data_dir: Path, label: str, checkpoint: Path) -> Path:
    return data_dir / f"no_leak_{label}_{checkpoint_cache_key(checkpoint)}_scores.npy"


def predict_or_load(
    data_dir: Path,
    label: str,
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    cache = prediction_cache_path(data_dir, label, checkpoint)
    if cache.exists():
        return np.asarray(np.load(cache, mmap_mode="r"), dtype=np.float32)
    model = ActionValueSetReranker.from_checkpoint(checkpoint, map_location=device).to(device)
    model.eval()
    states = np.load(data_dir / "states.npy", mmap_mode="r")
    group_ids = np.asarray(np.load(data_dir / "group_ids.npy", mmap_mode="r"), dtype=np.int64)
    bounds = build_group_bounds(group_ids)
    dataset = CandidateGroupDataset(data_dir, list(range(len(bounds))), bounds, model.score_mean, model.score_std)
    if int(dataset.input_dim) != int(model.input_dim):
        raise ValueError(
            f"{data_dir}: dataset input_dim={dataset.input_dim} does not match "
            f"{checkpoint} input_dim={model.input_dim}"
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_groups,
    )
    scores = predict_groups(model, loader, bounds, int(states.shape[0]), device).astype(np.float32)
    np.save(cache, scores)
    return scores


def compact_metrics(metrics: dict) -> dict:
    keys = [
        "groups",
        "samples",
        "group_top1",
        "group_top3",
        "group_top5",
        "group_top10",
        "group_top15",
        "group_top20",
        "group_top1_regret",
        "group_top3_rerank_regret",
        "group_top10_rerank_regret",
    ]
    return {key: metrics[key] for key in keys if key in metrics}


def evaluate_dataset(
    name: str,
    dim520: Path,
    dim617: Path,
    old_checkpoints: list[Path],
    new_checkpoints: list[Path],
    blends: list[tuple[str, np.ndarray]],
    gammas: list[float],
    topks: list[int],
    device: torch.device,
    batch_size: int,
) -> dict:
    scores, _, bounds, base_scores = load_arrays(dim617)
    old_scores = [
        predict_or_load(dim520, f"old{i + 1}_520", checkpoint, device, batch_size)
        for i, checkpoint in enumerate(old_checkpoints)
    ]
    new_scores = [
        predict_or_load(dim617, f"new{i + 1}_617", checkpoint, device, batch_size)
        for i, checkpoint in enumerate(new_checkpoints)
    ]
    model_scores = old_scores + new_scores
    if len(model_scores) != 4:
        raise ValueError("Expected exactly two old and two new checkpoints")

    base_metrics, _ = summarize_groups(scores, base_scores, bounds, topks)
    rows: list[dict] = [
        {
            "selector": "base_scores",
            "gamma": 0.0,
            "metrics": compact_metrics(base_metrics),
        }
    ]
    for blend_name, weights in blends:
        blended_model = np.zeros_like(base_scores, dtype=np.float32)
        for weight, model_score in zip(weights, model_scores):
            blended_model += float(weight) * model_score
        for gamma in gammas:
            pred = base_scores + float(gamma) * (blended_model - base_scores)
            metrics, _ = summarize_groups(scores, pred.astype(np.float32), bounds, topks)
            rows.append(
                {
                    "selector": blend_name,
                    "gamma": float(gamma),
                    "weights": [float(x) for x in weights],
                    "metrics": compact_metrics(metrics),
                }
            )
    rows.sort(
        key=lambda row: (
            float(row["metrics"].get("group_top1", 0.0)),
            -float(row["metrics"].get("group_top1_regret", 0.0)),
            float(row["metrics"].get("group_top3", 0.0)),
            -float(row["metrics"].get("group_top10_rerank_regret", 0.0)),
        ),
        reverse=True,
    )
    return {
        "name": name,
        "dim520": str(dim520),
        "dim617": str(dim617),
        "rows": rows,
    }


def aggregate(rows_by_dataset: dict[str, dict], top_n: int = 20) -> list[dict]:
    selector_keys: set[tuple[str, float]] = set()
    for dataset in rows_by_dataset.values():
        for row in dataset["rows"]:
            selector_keys.add((str(row["selector"]), float(row["gamma"])))
    output: list[dict] = []
    for selector, gamma in selector_keys:
        matched = []
        for name, dataset in rows_by_dataset.items():
            found = next(
                (
                    row
                    for row in dataset["rows"]
                    if str(row["selector"]) == selector and abs(float(row["gamma"]) - gamma) < 1e-9
                ),
                None,
            )
            if found is not None:
                matched.append((name, found["metrics"]))
        if not matched:
            continue
        n = len(matched)
        avg_top1 = sum(float(metrics.get("group_top1", 0.0)) for _, metrics in matched) / n
        avg_reg1 = sum(float(metrics.get("group_top1_regret", 0.0)) for _, metrics in matched) / n
        min_top1 = min(float(metrics.get("group_top1", 0.0)) for _, metrics in matched)
        avg_reg10 = sum(float(metrics.get("group_top10_rerank_regret", 0.0)) for _, metrics in matched) / n
        output.append(
            {
                "selector": selector,
                "gamma": gamma,
                "avg_top1": avg_top1,
                "min_top1": min_top1,
                "avg_reg1": avg_reg1,
                "avg_reg10": avg_reg10,
                "datasets": {name: metrics for name, metrics in matched},
            }
        )
    output.sort(key=lambda row: (row["avg_top1"], -row["avg_reg1"], row["min_top1"]), reverse=True)
    return output[:top_n]


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# T2 No-Leak Set Blend Evaluation",
        "",
        f"- elapsed_seconds: {summary['elapsed_seconds']:.1f}",
        f"- device: `{summary['device']}`",
        "- no candidate_ranks.npy or route_tags.npy are used.",
        "",
        "## Aggregate",
        "",
        "| selector | gamma | avg Top1 | min Top1 | avg Reg1 | avg Reg10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["aggregate"]:
        lines.append(
            f"| {row['selector']} | {row['gamma']:.2f} | {row['avg_top1']:.1%} | "
            f"{row['min_top1']:.1%} | {row['avg_reg1']:.3f} | {row['avg_reg10']:.3f} |"
        )
    for dataset in summary["datasets"]:
        lines.extend(
            [
                "",
                f"## {dataset['name']}",
                "",
                "| selector | gamma | Top1 | Top3 | Top5 | Top10 | Top20 | Reg1 | Reg3 | Reg10 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in dataset["rows"][:15]:
            m = row["metrics"]
            lines.append(
                f"| {row['selector']} | {float(row['gamma']):.2f} | "
                f"{float(m.get('group_top1', 0.0)):.1%} | "
                f"{float(m.get('group_top3', 0.0)):.1%} | "
                f"{float(m.get('group_top5', 0.0)):.1%} | "
                f"{float(m.get('group_top10', 0.0)):.1%} | "
                f"{float(m.get('group_top20', 0.0)):.1%} | "
                f"{float(m.get('group_top1_regret', 0.0)):.3f} | "
                f"{float(m.get('group_top3_rerank_regret', 0.0)):.3f} | "
                f"{float(m.get('group_top10_rerank_regret', 0.0)):.3f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, help="name=dim520,dim617")
    parser.add_argument("--old-checkpoints", nargs=2, required=True)
    parser.add_argument("--new-checkpoints", nargs=2, required=True)
    parser.add_argument("--blend", action="append", default=[])
    parser.add_argument("--gammas", default="0,0.25,0.5,0.75,1.0,1.15,1.25")
    parser.add_argument("--topks", default="1,3,5,10,15,20")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    started = time.time()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    datasets = [parse_dataset(spec) for spec in args.dataset]
    blends = [parse_blend(spec) for spec in args.blend]
    if not blends:
        blends = [
            ("old_avg", np.asarray([0.5, 0.5, 0.0, 0.0], dtype=np.float32)),
            ("new_avg", np.asarray([0.0, 0.0, 0.5, 0.5], dtype=np.float32)),
            ("old_new_avg", np.asarray([0.25, 0.25, 0.25, 0.25], dtype=np.float32)),
            ("old_r2", np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)),
            ("new_r2", np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)),
        ]
    gammas = [float(part) for part in args.gammas.split(",") if part.strip()]
    topks = parse_topks(args.topks)
    old_checkpoints = [Path(path) for path in args.old_checkpoints]
    new_checkpoints = [Path(path) for path in args.new_checkpoints]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_by_dataset: dict[str, dict] = {}
    for name, dim520, dim617 in datasets:
        rows_by_dataset[name] = evaluate_dataset(
            name,
            dim520,
            dim617,
            old_checkpoints,
            new_checkpoints,
            blends,
            gammas,
            topks,
            device,
            args.batch_size,
        )

    summary = {
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        "old_checkpoints": [str(path) for path in old_checkpoints],
        "new_checkpoints": [str(path) for path in new_checkpoints],
        "gammas": gammas,
        "blends": [{"name": name, "weights": [float(x) for x in weights]} for name, weights in blends],
        "datasets": list(rows_by_dataset.values()),
        "aggregate": aggregate(rows_by_dataset),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_markdown(out_dir / "summary.md", summary)
    print(json.dumps({"summary": str(out_dir / "summary.json"), "elapsed_seconds": summary["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
