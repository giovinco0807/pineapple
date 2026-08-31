"""Create weighted action-value data from evaluation misses.

``evaluate_action_value_reranker`` writes one row per group whose teacher-best
action is outside the requested Top-K.  This utility turns those rows back into
sample/group weights for the same data directory, so fine-tuning can focus on
the exact groups that the current model or ensemble failed to keep.
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
    "base_scores.npy",
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


def load_miss_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    row["_misses_path"] = str(path)
                    rows.append(row)
    return rows


def link_or_copy(src: Path, dst: Path) -> str:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def boost_row(
    row: dict[str, Any],
    bounds: list[tuple[int, int]],
    scores: np.ndarray,
    sample_weights: np.ndarray,
    group_weights: np.ndarray,
    base_weights: np.ndarray,
    teacher_best_add: float,
    confuser_add: float,
    hard_group_add: float,
    regret_group_scale: float,
    regret_cap: float,
    min_regret: float,
    top_confusers: int,
) -> tuple[bool, int]:
    group_index = int(row["group_id"])
    if group_index < 0 or group_index >= len(bounds):
        raise ValueError(f"group_id={group_index} is outside source bounds")
    regret = max(0.0, float(row.get("regret", 0.0)))
    if regret < min_regret:
        return False, 0

    start, end = bounds[group_index]
    idx = np.arange(start, end, dtype=np.int64)
    teacher_idx = int(idx[int(np.argmax(np.asarray(scores[idx], dtype=np.float64)))])

    group_weights[group_index] = max(
        float(group_weights[group_index]),
        1.0 + float(hard_group_add) + min(regret, regret_cap) * float(regret_group_scale),
    )
    sample_weights[teacher_idx] += float(teacher_best_add)

    predictions = row.get("top_predictions") or [
        {"sample_index": int(sample_index)}
        for sample_index in (row.get("ordered_indices") or [])
    ]
    boosted_confusers = 0
    for pred in predictions[:top_confusers]:
        confuser_idx = int(pred.get("sample_index", -1))
        if confuser_idx == teacher_idx or confuser_idx < start or confuser_idx >= end:
            continue
        sample_weights[confuser_idx] += float(confuser_add)
        boosted_confusers += 1

    return bool(np.any(sample_weights[idx] > base_weights[idx])), boosted_confusers


def create(args: argparse.Namespace) -> None:
    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    meta = load_metadata(source)

    group_ids_mm = np.load(source / "group_ids.npy", mmap_mode="r")
    n_samples = min(int(meta.get("n_samples", group_ids_mm.shape[0])), int(group_ids_mm.shape[0]))
    group_ids = np.asarray(group_ids_mm[:n_samples], dtype=np.int64)
    scores = np.asarray(np.load(source / "scores.npy", mmap_mode="r")[:n_samples], dtype=np.float32)
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

    rows = load_miss_rows(Path(path) for path in args.misses)
    stats: dict[str, Any] = {
        "groups": int(len(bounds)),
        "rows": int(len(rows)),
        "weighted_groups": 0,
        "weighted_candidates": 0,
        "confusers_boosted": 0,
        "min_regret": float(args.min_regret),
    }
    weighted_groups: set[int] = set()

    for row in rows:
        before = sample_weights.copy() if args.debug_copy_weights else None
        weighted, confusers = boost_row(
            row,
            bounds,
            scores,
            sample_weights,
            group_weights,
            base_weights,
            args.teacher_best_add,
            args.confuser_add,
            args.hard_group_add,
            args.regret_group_scale,
            args.regret_cap,
            args.min_regret,
            args.top_confusers,
        )
        if weighted:
            group_index = int(row["group_id"])
            weighted_groups.add(group_index)
            start, end = bounds[group_index]
            if before is None:
                stats["weighted_candidates"] += int(np.sum(sample_weights[start:end] > base_weights[start:end]))
            else:
                stats["weighted_candidates"] += int(np.sum(sample_weights[start:end] > before[start:end]))
            stats["confusers_boosted"] += confusers

    stats["weighted_groups"] = int(len(weighted_groups))
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
            "eval_miss_rows": [str(path) for path in args.misses],
            "eval_miss_weighted_stats": stats,
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
    parser = argparse.ArgumentParser(description="Create weighted action-value data from evaluator miss rows")
    parser.add_argument("--source", required=True)
    parser.add_argument("--misses", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-regret", type=float, default=0.0)
    parser.add_argument("--teacher-best-add", type=float, default=8.0)
    parser.add_argument("--confuser-add", type=float, default=1.5)
    parser.add_argument("--top-confusers", type=int, default=5)
    parser.add_argument("--hard-group-add", type=float, default=8.0)
    parser.add_argument("--regret-group-scale", type=float, default=1.5)
    parser.add_argument("--regret-cap", type=float, default=8.0)
    parser.add_argument("--max-sample-weight", type=float, default=24.0)
    parser.add_argument("--max-group-weight", type=float, default=24.0)
    parser.add_argument("--debug-copy-weights", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    create(args)


if __name__ == "__main__":
    main()
