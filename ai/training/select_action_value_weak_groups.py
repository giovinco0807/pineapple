"""Select complete action-value decision groups from a weak-groups report.

The evaluator writes one row per decision group into ``weak_groups.jsonl``.
This utility copies the matching complete candidate groups from the original
action-value dataset, preserving all candidate-level labels so the subset can
be used as an extra training source or as a focused stress set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


BASE_ARRAYS = (
    "states",
    "scores",
    "bust",
    "fl",
    "fl_types",
    "turns",
    "action_indices",
    "candidate_ranks",
    "route_tags",
    "sample_weights",
    "teacher_gaps",
    "positions",
)


def load_meta(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "metadata.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def parse_turns(raw: str) -> set[int] | None:
    if not raw:
        return None
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def topk_regret(item: dict[str, Any], topk: int) -> float:
    regrets = item.get("topk_exact_rerank_regret") or {}
    return float(regrets.get(str(topk), regrets.get(topk, 0.0)))


def load_selected_weak_groups(args: argparse.Namespace) -> list[dict[str, Any]]:
    turns = parse_turns(args.turns)
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    with Path(args.weak_groups).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            gid = int(item["group_id"])
            if gid in seen:
                continue
            if turns is not None and int(item.get("turn", -1)) not in turns:
                continue
            if args.min_rank and int(item.get("rank", 0)) <= args.min_rank:
                continue
            if args.min_regret and float(item.get("regret", 0.0)) < args.min_regret:
                continue
            if args.min_topk_regret and topk_regret(item, args.topk) < args.min_topk_regret:
                continue
            selected.append(item)
            seen.add(gid)
            if args.max_groups and len(selected) >= args.max_groups:
                break
    selected.sort(key=lambda item: int(item["group_id"]))
    return selected


def group_bounds(data_dir: Path) -> dict[int, tuple[int, int]]:
    meta = load_meta(data_dir)
    group_ids_mm = np.load(data_dir / "group_ids.npy", mmap_mode="r")
    n_samples = min(int(meta.get("n_samples", group_ids_mm.shape[0])), group_ids_mm.shape[0])
    group_ids = np.asarray(group_ids_mm[:n_samples], dtype=np.int64)
    bounds: dict[int, tuple[int, int]] = {}
    if len(group_ids) == 0:
        return bounds
    start = 0
    current = int(group_ids[0])
    for idx in range(1, len(group_ids)):
        gid = int(group_ids[idx])
        if gid != current:
            bounds[current] = (start, idx)
            start = idx
            current = gid
    bounds[current] = (start, len(group_ids))
    return bounds


def available_arrays(data_dir: Path) -> list[str]:
    return [name for name in BASE_ARRAYS if (data_dir / f"{name}.npy").exists()]


def allocate_outputs(data_dir: Path, output_dir: Path, names: list[str], total: int) -> dict[str, np.ndarray]:
    outputs: dict[str, np.ndarray] = {}
    for name in names:
        src = np.load(data_dir / f"{name}.npy", mmap_mode="r")
        shape = (total,) if src.ndim == 1 else (total, *src.shape[1:])
        outputs[name] = np.lib.format.open_memmap(
            output_dir / f"{name}.npy",
            mode="w+",
            dtype=src.dtype,
            shape=shape,
        )
    return outputs


def copy_groups(args: argparse.Namespace, selected: list[dict[str, Any]]) -> dict[str, Any]:
    data_dir = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    bounds = group_bounds(data_dir)
    missing = [int(item["group_id"]) for item in selected if int(item["group_id"]) not in bounds]
    if missing:
        raise SystemExit(f"{len(missing)} selected groups are missing from source data; first={missing[:5]}")

    ranges = [(bounds[int(item["group_id"])][0], bounds[int(item["group_id"])][1], item) for item in selected]
    total = sum(end - start for start, end, _ in ranges)
    names = available_arrays(data_dir)
    outputs = allocate_outputs(data_dir, output_dir, names, total)
    group_ids_out = np.lib.format.open_memmap(output_dir / "group_ids.npy", mode="w+", dtype=np.int64, shape=(total,))
    arrays = {name: np.load(data_dir / f"{name}.npy", mmap_mode="r") for name in names}

    pos = 0
    selected_rows: list[dict[str, Any]] = []
    for new_gid, (start, end, item) in enumerate(ranges):
        length = end - start
        dst = slice(pos, pos + length)
        for name, arr in arrays.items():
            outputs[name][dst] = arr[start:end]
        group_ids_out[dst] = new_gid
        row = dict(item)
        row["source_group_id"] = int(item["group_id"])
        row["group_id"] = new_gid
        row["source_start"] = int(start)
        row["source_end"] = int(end)
        row["n_copied_candidates"] = int(length)
        selected_rows.append(row)
        pos += length

    for arr in list(outputs.values()) + [group_ids_out]:
        arr.flush()

    with (output_dir / "selected_groups.jsonl").open("w", encoding="utf-8") as f:
        for row in selected_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    turns = np.load(output_dir / "turns.npy", mmap_mode="r") if (output_dir / "turns.npy").exists() else np.array([])
    sample_weights = (
        np.load(output_dir / "sample_weights.npy", mmap_mode="r")
        if (output_dir / "sample_weights.npy").exists()
        else np.array([], dtype=np.float32)
    )
    meta = {
        "source_data": str(data_dir),
        "weak_groups": str(Path(args.weak_groups)),
        "n_samples": int(total),
        "n_groups": int(len(selected_rows)),
        "arrays": names,
        "selection": {
            "turns": args.turns,
            "topk": int(args.topk),
            "min_rank": int(args.min_rank),
            "min_regret": float(args.min_regret),
            "min_topk_regret": float(args.min_topk_regret),
            "max_groups": int(args.max_groups),
        },
        "turns": {str(int(t)): int((turns == t).sum()) for t in sorted(set(np.asarray(turns).tolist()))},
        "sample_weight_mean": float(np.asarray(sample_weights).mean()) if len(sample_weights) else 1.0,
        "sample_weight_max": float(np.asarray(sample_weights).max()) if len(sample_weights) else 1.0,
        "source_rank_max": max((int(row.get("rank", 0)) for row in selected_rows), default=0),
        "source_regret_max": max((float(row.get("regret", 0.0)) for row in selected_rows), default=0.0),
        f"source_top{args.topk}_regret_max": max((topk_regret(row, args.topk) for row in selected_rows), default=0.0),
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Copy selected action-value weak groups into a new data directory")
    parser.add_argument("--data", required=True, help="Source action-value data directory")
    parser.add_argument("--weak-groups", required=True, help="weak_groups.jsonl produced by evaluate_action_value_dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--turns", default="", help="Comma-separated turn ids to keep, e.g. 2 or 2,3")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--min-rank", type=int, default=0, help="Keep groups with rank greater than this value")
    parser.add_argument("--min-regret", type=float, default=0.0)
    parser.add_argument("--min-topk-regret", type=float, default=0.0)
    parser.add_argument("--max-groups", type=int, default=0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    selected = load_selected_weak_groups(args)
    if not selected:
        raise SystemExit("No weak groups matched the filters.")
    meta = copy_groups(args, selected)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
