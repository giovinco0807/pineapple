"""Merge candidate-level action-value reranker data directories."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


ARRAY_SPECS = {
    "states": np.float32,
    "scores": np.float32,
    "bust": np.float32,
    "fl": np.float32,
    "fl_types": np.float32,
    "turns": np.int16,
    "action_indices": np.int16,
    "candidate_ranks": np.int16,
    "group_ids": np.int64,
    "route_tags": np.int16,
    "base_scores": np.float32,
    "sample_weights": np.float32,
    "teacher_gaps": np.float32,
    "positions": np.int8,
}

GROUP_ARRAY_SPECS = {
    "group_sample_weights": np.float32,
}


def load_metadata(data_dir: Path) -> dict:
    path = data_dir / "metadata.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sample_count(data_dir: Path) -> int:
    meta = load_metadata(data_dir)
    states = np.load(data_dir / "states.npy", mmap_mode="r")
    return min(int(meta.get("n_samples", states.shape[0])), int(states.shape[0]))


def group_count(data_dir: Path, n_samples: int) -> int:
    if n_samples <= 0:
        return 0
    group_ids = np.asarray(np.load(data_dir / "group_ids.npy", mmap_mode="r")[:n_samples])
    return int(len(np.unique(group_ids)))


def first_shape(data_dirs: list[Path], name: str) -> tuple[int, ...]:
    for data_dir in data_dirs:
        path = data_dir / f"{name}.npy"
        if path.exists():
            arr = np.load(path, mmap_mode="r")
            return tuple(arr.shape[1:])
    if name in {"base_scores", "positions"}:
        return ()
    raise FileNotFoundError(f"No input has {name}.npy")


def merge(args: argparse.Namespace) -> dict:
    start = time.time()
    data_dirs = [Path(d) for d in args.dirs]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = [sample_count(d) for d in data_dirs]
    groups = [group_count(d, n) for d, n in zip(data_dirs, counts)]
    total = int(sum(counts))
    if total <= 0:
        raise SystemExit("No samples to merge.")

    print("Merging action-value reranker data")
    for data_dir, n, g in zip(data_dirs, counts, groups):
        print(f"  {data_dir}: samples={n:,} groups={g:,}")
    print(f"  output: {out_dir}")
    print(f"  total samples: {total:,}")

    out_arrays = {}
    for name, dtype in ARRAY_SPECS.items():
        shape_tail = first_shape(data_dirs, name)
        out_arrays[name] = np.lib.format.open_memmap(
            out_dir / f"{name}.npy",
            mode="w+",
            dtype=dtype,
            shape=(total, *shape_tail),
        )
    out_group_arrays = {}
    total_groups = int(sum(groups))
    for name, dtype in GROUP_ARRAY_SPECS.items():
        out_group_arrays[name] = np.lib.format.open_memmap(
            out_dir / f"{name}.npy",
            mode="w+",
            dtype=dtype,
            shape=(total_groups,),
        )

    offset = 0
    group_offset = 0
    turns: dict[str, int] = {}
    skipped = 0
    for data_dir, n in zip(data_dirs, counts):
        end = offset + n
        for name in ARRAY_SPECS:
            path = data_dir / f"{name}.npy"
            if path.exists():
                arr = np.load(path, mmap_mode="r")[:n]
            elif name == "base_scores":
                arr = np.zeros((n,), dtype=ARRAY_SPECS[name])
            elif name == "positions":
                arr = np.zeros((n,), dtype=ARRAY_SPECS[name])
            else:
                raise FileNotFoundError(path)
            if name == "group_ids":
                out_arrays[name][offset:end] = np.asarray(arr, dtype=np.int64) + group_offset
            else:
                out_arrays[name][offset:end] = arr
        group_end = group_offset + group_count(data_dir, n)
        for name, dtype in GROUP_ARRAY_SPECS.items():
            path = data_dir / f"{name}.npy"
            if path.exists():
                arr = np.load(path, mmap_mode="r")[: group_end - group_offset]
            else:
                arr = np.ones((group_end - group_offset,), dtype=dtype)
            out_group_arrays[name][group_offset:group_end] = arr
        turn_arr = np.asarray(np.load(data_dir / "turns.npy", mmap_mode="r")[:n])
        for turn in np.unique(turn_arr):
            turns[str(int(turn))] = turns.get(str(int(turn)), 0) + int((turn_arr == turn).sum())
        meta = load_metadata(data_dir)
        skipped += int(meta.get("skipped", 0))
        group_offset = group_end
        offset = end

    for arr in out_arrays.values():
        arr.flush()
    for arr in out_group_arrays.values():
        arr.flush()

    scores = np.asarray(out_arrays["scores"][:total])
    bust = np.asarray(out_arrays["bust"][:total])
    fl = np.asarray(out_arrays["fl"][:total])
    sample_weights = np.asarray(out_arrays["sample_weights"][:total])
    group_sample_weights = np.asarray(out_group_arrays["group_sample_weights"][:group_offset])
    route_tags = np.asarray(out_arrays["route_tags"][:total])
    route_tag_bits = {}
    for data_dir in data_dirs:
        route_tag_bits = load_metadata(data_dir).get("route_tag_bits", route_tag_bits)
        if route_tag_bits:
            break
    route_counts = {}
    for name, bit in route_tag_bits.items():
        route_counts[name] = int((route_tags & int(bit) != 0).sum())

    metadata = {
        "source_dirs": [str(d) for d in data_dirs],
        "source_samples": [int(n) for n in counts],
        "source_groups": [int(g) for g in groups],
        "n_samples": int(total),
        "n_records": int(group_offset),
        "state_dim": int(out_arrays["states"].shape[1]),
        "turns": dict(sorted(turns.items())),
        "skipped": int(skipped),
        "score_mean": float(scores.mean()) if total else 0.0,
        "score_std": float(scores.std()) if total else 1.0,
        "bust_mean": float(bust.mean()) if total else 0.0,
        "fl_mean": float(fl.mean()) if total else 0.0,
        "sample_weight_mean": float(sample_weights.mean()) if total else 1.0,
        "sample_weight_max": float(sample_weights.max()) if total else 1.0,
        "group_sample_weight_mean": float(group_sample_weights.mean()) if group_offset else 1.0,
        "group_sample_weight_max": float(group_sample_weights.max()) if group_offset else 1.0,
        "route_tag_bits": route_tag_bits,
        "route_counts": route_counts,
        "elapsed_seconds": time.time() - start,
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  groups={group_offset:,} score={metadata['score_mean']:+.3f} +/- {metadata['score_std']:.3f}")
    return metadata


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Merge action-value reranker data directories")
    parser.add_argument("--dirs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(merge(args), indent=2))


if __name__ == "__main__":
    main()
