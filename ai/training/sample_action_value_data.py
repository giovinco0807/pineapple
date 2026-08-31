"""Build a mixed action-value dataset by sampling complete decision groups."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


ARRAY_SPECS = {
    "states": np.float32,
    "scores": np.float32,
    "bust": np.float32,
    "fl": np.float32,
    "fl_types": np.float32,
    "turns": np.int16,
    "action_indices": np.int16,
    "candidate_ranks": np.int16,
    "route_tags": np.int16,
    "base_scores": np.float32,
    "sample_weights": np.float32,
    "teacher_gaps": np.float32,
}


def labels_available(data_dir: Path, override: str) -> bool:
    if override == "true":
        return True
    if override == "false":
        return False
    meta = load_meta(data_dir)
    label_note = str(meta.get("label_note", "")).lower()
    if "unavailable" in label_note or "set to zero" in label_note:
        return False
    return True


def load_meta(data_dir: Path) -> dict:
    path = data_dir / "metadata.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def group_bounds(data_dir: Path, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    gids = np.asarray(np.load(data_dir / "group_ids.npy", mmap_mode="r")[:n_samples])
    if len(gids) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    starts = [0]
    for i in range(1, len(gids)):
        if gids[i] != gids[i - 1]:
            starts.append(i)
    starts_arr = np.asarray(starts, dtype=np.int64)
    ends_arr = np.concatenate([starts_arr[1:], np.asarray([len(gids)], dtype=np.int64)])
    return starts_arr, ends_arr


def ranges_for_group_ids(data_dir: Path, requested_group_ids: Iterable[int]) -> tuple[list[tuple[int, int]], int]:
    states = np.load(data_dir / "states.npy", mmap_mode="r")
    meta = load_meta(data_dir)
    n_samples = min(int(meta.get("n_samples", states.shape[0])), states.shape[0])
    starts, ends = group_bounds(data_dir, n_samples)
    group_ids = np.asarray(np.load(data_dir / "group_ids.npy", mmap_mode="r")[:n_samples])
    requested = {int(group_id) for group_id in requested_group_ids}
    ranges: list[tuple[int, int]] = []
    selected = 0
    for start, end in zip(starts, ends):
        if int(group_ids[int(start)]) not in requested:
            continue
        start_i = int(start)
        end_i = int(end)
        ranges.append((start_i, end_i))
        selected += end_i - start_i
    return ranges, selected


def hard_groups_from_jsonl(paths: Iterable[str]) -> dict[Path, set[int]]:
    groups: dict[Path, set[int]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                data = row.get("data")
                group_id = row.get("group_id")
                if data is None or group_id is None:
                    continue
                groups.setdefault(Path(data), set()).add(int(group_id))
    return groups


def select_group_ranges(data_dir: Path, max_samples: int, seed: int) -> tuple[list[tuple[int, int]], int]:
    states = np.load(data_dir / "states.npy", mmap_mode="r")
    meta = load_meta(data_dir)
    n_samples = min(int(meta.get("n_samples", states.shape[0])), states.shape[0])
    starts, ends = group_bounds(data_dir, n_samples)
    group_ids = np.arange(len(starts), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(group_ids)

    ranges: list[tuple[int, int]] = []
    selected = 0
    for gid in group_ids:
        start = int(starts[gid])
        end = int(ends[gid])
        ranges.append((start, end))
        selected += end - start
        if max_samples > 0 and selected >= max_samples:
            break
    ranges.sort()
    return ranges, selected


def all_group_ranges(data_dir: Path) -> tuple[list[tuple[int, int]], int]:
    states = np.load(data_dir / "states.npy", mmap_mode="r")
    meta = load_meta(data_dir)
    n_samples = min(int(meta.get("n_samples", states.shape[0])), states.shape[0])
    starts, ends = group_bounds(data_dir, n_samples)
    ranges = [(int(s), int(e)) for s, e in zip(starts, ends)]
    return ranges, n_samples


def source_plan(args: argparse.Namespace) -> list[dict]:
    plan = []
    if args.base_data:
        ranges, n = select_group_ranges(Path(args.base_data), args.base_samples, args.seed)
        data_dir = Path(args.base_data)
        plan.append({
            "role": "base",
            "data_dir": data_dir,
            "ranges": ranges,
            "samples": n,
            "aux_labels": labels_available(data_dir, args.base_aux_labels),
        })
    for idx, extra in enumerate(args.extra_data):
        extra_path = Path(extra)
        repeat = max(1, int(args.extra_repeat))
        for repeat_idx in range(repeat):
            if args.extra_samples > 0:
                ranges, n = select_group_ranges(
                    extra_path,
                    args.extra_samples,
                    args.seed + (idx * repeat) + repeat_idx + 1,
                )
            else:
                ranges, n = all_group_ranges(extra_path)
            plan.append(
                {
                    "role": "extra",
                    "data_dir": Path(extra),
                    "ranges": ranges,
                    "samples": n,
                    "repeat_index": repeat_idx,
                    "aux_labels": labels_available(extra_path, args.extra_aux_labels),
                }
            )
    hard_groups = hard_groups_from_jsonl(args.hard_group_jsonl)
    for hard_idx, (data_dir, group_ids) in enumerate(sorted(hard_groups.items(), key=lambda item: str(item[0]))):
        ranges, n = ranges_for_group_ids(data_dir, group_ids)
        repeat = max(1, int(args.hard_repeat))
        for repeat_idx in range(repeat):
            plan.append(
                {
                    "role": "hard",
                    "data_dir": data_dir,
                    "ranges": ranges,
                    "samples": n,
                    "repeat_index": repeat_idx,
                    "hard_source_index": hard_idx,
                    "aux_labels": labels_available(data_dir, args.hard_aux_labels),
                }
            )
    return plan


def shape_for_output(name: str, first_dir: Path, total: int) -> tuple[int, ...]:
    arr = np.load(first_dir / f"{name}.npy", mmap_mode="r")
    if arr.ndim == 1:
        return (total,)
    return (total, *arr.shape[1:])


def copy_ranges(plan: list[dict], out_dir: Path, total: int) -> dict:
    first_dir = plan[0]["data_dir"]
    outputs = {}
    for name, dtype in ARRAY_SPECS.items():
        outputs[name] = np.lib.format.open_memmap(
            out_dir / f"{name}.npy",
            mode="w+",
            dtype=dtype,
            shape=shape_for_output(name, first_dir, total),
        )
    group_ids_out = np.lib.format.open_memmap(
        out_dir / "group_ids.npy", mode="w+", dtype=np.int64, shape=(total,)
    )
    aux_label_mask_out = np.lib.format.open_memmap(
        out_dir / "aux_label_mask.npy", mode="w+", dtype=np.float32, shape=(total,)
    )

    pos = 0
    new_group = 0
    samples_by_role = {}
    groups_by_role = {}
    samples_by_aux_label = {"available": 0, "missing": 0}
    for item in plan:
        data_dir = item["data_dir"]
        arrays = {name: np.load(data_dir / f"{name}.npy", mmap_mode="r") for name in ARRAY_SPECS}
        aux_value = 1.0 if item.get("aux_labels", True) else 0.0
        role_samples = 0
        role_groups = 0
        for start, end in item["ranges"]:
            length = end - start
            dst = slice(pos, pos + length)
            for name, arr in arrays.items():
                outputs[name][dst] = arr[start:end]
            group_ids_out[dst] = new_group
            aux_label_mask_out[dst] = aux_value
            pos += length
            new_group += 1
            role_samples += length
            role_groups += 1
        samples_by_role[item["role"]] = samples_by_role.get(item["role"], 0) + role_samples
        groups_by_role[item["role"]] = groups_by_role.get(item["role"], 0) + role_groups
        aux_key = "available" if aux_value > 0.0 else "missing"
        samples_by_aux_label[aux_key] += role_samples

    for arr in list(outputs.values()) + [group_ids_out, aux_label_mask_out]:
        arr.flush()

    return {
        "n_samples": int(pos),
        "n_groups": int(new_group),
        "samples_by_role": samples_by_role,
        "groups_by_role": groups_by_role,
        "samples_by_aux_label": samples_by_aux_label,
    }


def build(args: argparse.Namespace) -> None:
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = source_plan(args)
    if not plan:
        raise SystemExit("No input data provided.")
    total = sum(int(item["samples"]) for item in plan)
    meta = copy_ranges(plan, out_dir, total)
    meta.update({
        "sources": [
            {
                "role": item["role"],
                "data_dir": str(item["data_dir"]),
                "selected_groups": len(item["ranges"]),
                "selected_samples": int(item["samples"]),
                "repeat_index": int(item.get("repeat_index", 0)),
                "hard_source_index": int(item.get("hard_source_index", -1)),
                "aux_labels": bool(item.get("aux_labels", True)),
            }
            for item in plan
        ],
        "base_samples_requested": int(args.base_samples),
        "extra_samples_requested": int(args.extra_samples),
        "extra_repeat": int(args.extra_repeat),
        "hard_group_jsonl": list(args.hard_group_jsonl),
        "hard_repeat": int(args.hard_repeat),
        "base_aux_labels": args.base_aux_labels,
        "extra_aux_labels": args.extra_aux_labels,
        "hard_aux_labels": args.hard_aux_labels,
        "seed": int(args.seed),
    })
    turns = np.load(out_dir / "turns.npy", mmap_mode="r")
    route_tags = np.load(out_dir / "route_tags.npy", mmap_mode="r")
    sample_weights = np.load(out_dir / "sample_weights.npy", mmap_mode="r")
    meta["turns"] = {str(t): int((turns == t).sum()) for t in sorted(set(np.asarray(turns).tolist()))}
    meta["sample_weight_mean"] = float(np.asarray(sample_weights).mean()) if len(sample_weights) else 1.0
    meta["sample_weight_max"] = float(np.asarray(sample_weights).max()) if len(sample_weights) else 1.0
    meta["route_tag_nonzero"] = int((np.asarray(route_tags) != 0).sum())
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote mixed action-value data: {out_dir}")
    print(json.dumps(meta, indent=2))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sample and merge action-value datasets")
    parser.add_argument("--base-data", default=None)
    parser.add_argument("--base-samples", type=int, default=300_000)
    parser.add_argument("--base-aux-labels", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--extra-data", action="append", default=[])
    parser.add_argument("--extra-aux-labels", choices=("auto", "true", "false"), default="auto")
    parser.add_argument(
        "--extra-samples",
        type=int,
        default=0,
        help="Sample this many rows from each extra dataset; default 0 keeps all extra rows.",
    )
    parser.add_argument(
        "--extra-repeat",
        type=int,
        default=1,
        help="Repeat each extra dataset this many times to upweight active-loop hard states.",
    )
    parser.add_argument(
        "--hard-group-jsonl",
        action="append",
        default=[],
        help="Audit JSONL containing data and group_id fields for hard groups to upweight.",
    )
    parser.add_argument(
        "--hard-repeat",
        type=int,
        default=1,
        help="Repeat hard groups this many times.",
    )
    parser.add_argument("--hard-aux-labels", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260520)
    args = parser.parse_args(argv)
    build(args)


if __name__ == "__main__":
    main()
