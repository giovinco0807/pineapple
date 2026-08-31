"""Create weighted action-value data from candidate-pool miss rows.

The candidate-pool evaluator writes one JSON object per decision group with
``pool_metrics`` for each K.  This utility keeps the source arrays as hardlinks
where possible and writes fresh ``sample_weights.npy`` and
``group_sample_weights.npy`` so later training can focus on groups where a
multi-model pool missed the teacher-best action.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np


LINKED_ARRAYS = (
    "action_indices.npy",
    "aux_label_mask.npy",
    "bust.npy",
    "candidate_ranks.npy",
    "fl.npy",
    "fl_types.npy",
    "group_ids.npy",
    "positions.npy",
    "route_tags.npy",
    "scores.npy",
    "states.npy",
    "teacher_gaps.npy",
    "turns.npy",
)


def load_metadata(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "metadata.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def build_group_bounds(group_ids: np.ndarray) -> list[tuple[int, int]]:
    if len(group_ids) == 0:
        return []
    bounds: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(group_ids)):
        if int(group_ids[i]) != int(group_ids[i - 1]):
            bounds.append((start, i))
            start = i
    bounds.append((start, len(group_ids)))
    return bounds


def link_or_copy(src: Path, dst: Path) -> str:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pool_metric(row: dict[str, Any], k: int) -> dict[str, Any]:
    metrics = row.get("pool_metrics") or {}
    metric = metrics.get(str(k))
    if metric is None:
        metric = metrics.get(k)
    if not isinstance(metric, dict):
        raise KeyError(f"row group_id={row.get('group_id')} has no pool metric for K={k}")
    return metric


def boost_miss_group(
    row: dict[str, Any],
    bounds: list[tuple[int, int]],
    sample_weights: np.ndarray,
    group_weights: np.ndarray,
    base_weights: np.ndarray,
    miss_k: int,
    confuser_k: int,
    teacher_best_add: float,
    confuser_add: float,
    hard_group_add: float,
    regret_group_scale: float,
    regret_cap: float,
    min_ev_loss: float,
) -> tuple[int, int]:
    group_index = int(row["group_id"])
    start, end = bounds[group_index]
    teacher_idx = int(row["teacher_best_index"])
    if teacher_idx < start or teacher_idx >= end:
        raise ValueError(f"teacher index {teacher_idx} is outside group {group_index} [{start}, {end})")

    metric = pool_metric(row, miss_k)
    regret = max(0.0, float(metric.get("ev_loss", metric.get("regret", 0.0))))
    if bool(metric.get("hit", False)) or regret < float(min_ev_loss):
        return 0, 0

    group_weights[group_index] = max(
        float(group_weights[group_index]),
        1.0 + hard_group_add + min(regret, regret_cap) * regret_group_scale,
    )
    sample_weights[teacher_idx] += float(teacher_best_add)
    weighted = 1
    boosted_confusers = 0

    confuser_metric = pool_metric(row, confuser_k)
    for idx_raw in confuser_metric.get("indices") or []:
        idx = int(idx_raw)
        if idx == teacher_idx or idx < start or idx >= end:
            continue
        sample_weights[idx] += float(confuser_add)
        weighted += 1
        boosted_confusers += 1

    return weighted, boosted_confusers


def create(args: argparse.Namespace) -> None:
    source = Path(args.source)
    rows_path = Path(args.pool_rows)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    meta = load_metadata(source)
    group_ids_mm = np.load(source / "group_ids.npy", mmap_mode="r")
    n_samples = min(int(meta.get("n_samples", group_ids_mm.shape[0])), int(group_ids_mm.shape[0]))
    group_ids = np.asarray(group_ids_mm[:n_samples], dtype=np.int64)
    bounds = build_group_bounds(group_ids)
    base_weights = (
        np.asarray(np.load(source / "sample_weights.npy", mmap_mode="r")[:n_samples], dtype=np.float32)
        if (source / "sample_weights.npy").exists()
        else np.ones(n_samples, dtype=np.float32)
    )
    sample_weights = base_weights.copy()
    group_weights = (
        np.asarray(np.load(source / "group_sample_weights.npy", mmap_mode="r")[: len(bounds)], dtype=np.float32).copy()
        if (source / "group_sample_weights.npy").exists()
        else np.ones(len(bounds), dtype=np.float32)
    )
    rows = load_rows(rows_path)

    stats: dict[str, Any] = {
        "groups": int(len(bounds)),
        "rows": int(len(rows)),
        "primary_miss_k": int(args.miss_k),
        "secondary_miss_k": int(args.secondary_miss_k),
        "primary_miss_groups": 0,
        "secondary_miss_groups": 0,
        "weighted_candidates": 0,
        "confusers_boosted": 0,
        "min_ev_loss": float(args.min_ev_loss),
        "secondary_min_ev_loss": float(args.secondary_min_ev_loss),
    }

    for row in rows:
        if int(row["group_id"]) >= len(bounds):
            raise ValueError(f"row group_id={row['group_id']} is outside source bounds")
        weighted, confusers = boost_miss_group(
            row,
            bounds,
            sample_weights,
            group_weights,
            base_weights,
            args.miss_k,
            args.confuser_k,
            args.teacher_best_add,
            args.confuser_add,
            args.hard_group_add,
            args.regret_group_scale,
            args.regret_cap,
            args.min_ev_loss,
        )
        if weighted:
            stats["primary_miss_groups"] += 1
            stats["weighted_candidates"] += weighted
            stats["confusers_boosted"] += confusers
            continue

        if args.secondary_miss_k > 0:
            weighted, confusers = boost_miss_group(
                row,
                bounds,
                sample_weights,
                group_weights,
                base_weights,
                args.secondary_miss_k,
                args.confuser_k,
                args.secondary_teacher_best_add,
                args.secondary_confuser_add,
                args.secondary_hard_group_add,
                args.regret_group_scale,
                args.regret_cap,
                args.secondary_min_ev_loss,
            )
            if weighted:
                stats["secondary_miss_groups"] += 1
                stats["weighted_candidates"] += weighted
                stats["confusers_boosted"] += confusers

    sample_weights = np.clip(sample_weights, 0.01, float(args.max_sample_weight)).astype(np.float32)
    group_weights = np.clip(group_weights, 1.0, float(args.max_group_weight)).astype(np.float32)
    np.save(output / "sample_weights.npy", sample_weights)
    np.save(output / "group_sample_weights.npy", group_weights)

    link_modes: dict[str, str] = {}
    for name in LINKED_ARRAYS:
        src = source / name
        if src.exists():
            link_modes[name] = link_or_copy(src, output / name)

    new_meta = dict(meta)
    new_meta.update(
        {
            "source": str(source),
            "pool_rows": str(rows_path),
            "pool_miss_weighted_stats": stats,
            "group_sample_weight_mean": float(group_weights.mean()) if len(group_weights) else 0.0,
            "group_sample_weight_max": float(group_weights.max()) if len(group_weights) else 0.0,
            "sample_weight_mean": float(sample_weights.mean()) if len(sample_weights) else 0.0,
            "sample_weight_max": float(sample_weights.max()) if len(sample_weights) else 0.0,
            "link_modes": link_modes,
        }
    )
    with (output / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(new_meta, f, indent=2)
    print(json.dumps(new_meta, indent=2))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create weighted action-value data from candidate-pool misses")
    parser.add_argument("--source", required=True)
    parser.add_argument("--pool-rows", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--miss-k", type=int, default=5)
    parser.add_argument("--secondary-miss-k", type=int, default=3)
    parser.add_argument("--confuser-k", type=int, default=5)
    parser.add_argument("--min-ev-loss", type=float, default=0.0)
    parser.add_argument("--secondary-min-ev-loss", type=float, default=0.0)
    parser.add_argument("--teacher-best-add", type=float, default=12.0)
    parser.add_argument("--confuser-add", type=float, default=2.0)
    parser.add_argument("--hard-group-add", type=float, default=12.0)
    parser.add_argument("--secondary-teacher-best-add", type=float, default=5.0)
    parser.add_argument("--secondary-confuser-add", type=float, default=1.0)
    parser.add_argument("--secondary-hard-group-add", type=float, default=4.0)
    parser.add_argument("--regret-group-scale", type=float, default=2.0)
    parser.add_argument("--regret-cap", type=float, default=8.0)
    parser.add_argument("--max-sample-weight", type=float, default=24.0)
    parser.add_argument("--max-group-weight", type=float, default=24.0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    create(args)


if __name__ == "__main__":
    main()
