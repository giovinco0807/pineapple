"""Extract selected candidate groups from reranker feature datasets.

This is useful for hard-negative training: keep only the failed decision
groups, preserve their candidate rows, and remap them into a compact dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


INACTIVE_SELECTOR_SCORE = -1.0e7


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_named_path(value: str) -> tuple[str, Path]:
    name, sep, path = value.partition("=")
    if not sep or not name.strip() or not path.strip():
        raise ValueError("--data must be name=path")
    return name.strip(), Path(path.strip())


def group_slices(group_ids: np.ndarray) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    if len(group_ids) == 0:
        return out
    start = 0
    for i in range(1, len(group_ids)):
        if int(group_ids[i]) != int(group_ids[i - 1]):
            out[int(group_ids[start])] = (start, i)
            start = i
    out[int(group_ids[start])] = (start, len(group_ids))
    return out


def read_source_records(data_dir: Path) -> list[dict]:
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    source = Path(metadata["source"])
    records: list[dict] = []
    with source.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def copy_metadata(src: dict, n_samples: int, n_records: int, source_path: Path) -> dict:
    metadata = dict(src)
    metadata["source"] = str(source_path)
    metadata["n_allocated"] = int(n_samples)
    metadata["n_samples"] = int(n_samples)
    metadata["n_records"] = int(n_records)
    metadata["record_turns"] = {"2": int(n_records)}
    metadata["turns"] = {"2": int(n_samples)}
    metadata["note"] = "Extracted hard-negative reranker groups."
    return metadata


def run(args: argparse.Namespace) -> None:
    data_dirs = dict(parse_named_path(value) for value in args.data)
    selected_rows = load_jsonl(Path(args.rows))
    if args.model_label:
        selected_rows = [row for row in selected_rows if row.get("model_label") == args.model_label]
    if args.min_ev_loss > 0:
        selected_rows = [row for row in selected_rows if float(row.get("ev_loss", 0.0)) >= args.min_ev_loss]
    if not selected_rows:
        raise ValueError("no rows selected")

    grouped: dict[str, list[dict]] = {}
    for row in selected_rows:
        dataset = str(row["dataset"])
        if dataset not in data_dirs:
            raise KeyError(f"missing --data for dataset {dataset!r}")
        grouped.setdefault(dataset, []).append(row)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_out = out_dir / f"{args.name}.teacher.jsonl"
    source_records_out: list[dict] = []
    train_weight_rows: list[dict] = []

    arrays_by_name: dict[str, list[np.ndarray]] = {}
    selector_arrays_by_name: dict[str, list[np.ndarray]] = {}
    next_group_id = 0
    metadata_seed: dict | None = None

    for dataset in sorted(grouped):
        data_dir = data_dirs[dataset]
        metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
        if metadata_seed is None:
            metadata_seed = metadata
        group_ids = np.load(data_dir / "group_ids.npy")
        slices = group_slices(group_ids)
        source_records = read_source_records(data_dir)

        for row in sorted(grouped[dataset], key=lambda item: int(item["group_id"])):
            original_group_id = int(row["group_id"])
            if original_group_id not in slices:
                raise KeyError(f"{dataset}: group_id {original_group_id} not found")
            start, end = slices[original_group_id]
            for npy_path in sorted(data_dir.glob("*.npy")):
                values = np.load(npy_path)
                subset = np.asarray(values[start:end])
                if npy_path.name == "group_ids.npy":
                    subset = np.full((end - start,), next_group_id, dtype=values.dtype)
                arrays_by_name.setdefault(npy_path.name, []).append(subset)

            selector_dir = data_dir / "selector_scores"
            if selector_dir.exists():
                for npy_path in sorted(selector_dir.glob("*.npy")):
                    values = np.load(npy_path)
                    selector_arrays_by_name.setdefault(npy_path.name, []).append(
                        np.asarray(values[start:end])
                    )

            record = dict(source_records[original_group_id])
            record["_source_dataset"] = dataset
            record["_source_group_id"] = original_group_id
            source_records_out.append(record)
            train_weight_rows.append(
                {
                    "dataset": args.name,
                    "group_id": next_group_id,
                    "target_k": int(args.target_k),
                    "ev_loss": float(row.get("ev_loss", 0.0)),
                    "teacher_best_local_index": int(row.get("teacher_best_local_index", -1)),
                    "pred_local_index": int(row.get("pred_local_index", -1)),
                    "teacher_best_score": float(row.get("teacher_best_score", 0.0)),
                    "pred_score": float(row.get("pred_score", 0.0)),
                    "note": row.get("note")
                    or f"external1000 residual miss from {row.get('model_label')} {dataset}:{original_group_id}",
                }
            )
            next_group_id += 1

    write_jsonl(source_out, source_records_out)
    for name, chunks in arrays_by_name.items():
        np.save(out_dir / name, np.concatenate(chunks, axis=0))

    if selector_arrays_by_name:
        selector_dir = out_dir / "selector_scores"
        selector_dir.mkdir(exist_ok=True)
        total_samples = len(np.load(out_dir / "scores.npy"))
        for name, chunks in selector_arrays_by_name.items():
            combined = np.concatenate(chunks, axis=0)
            if len(combined) != total_samples:
                combined = np.full((total_samples,), INACTIVE_SELECTOR_SCORE, dtype=np.float32)
            np.save(selector_dir / name, combined)

    n_samples = int(len(np.load(out_dir / "scores.npy")))
    metadata = copy_metadata(metadata_seed or {}, n_samples, next_group_id, source_out)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    train_rows_path = out_dir / f"{args.name}.train_weight_rows.jsonl"
    write_jsonl(train_rows_path, train_weight_rows)
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "source": str(source_out),
                "train_weight_rows": str(train_rows_path),
                "records": int(next_group_id),
                "samples": int(n_samples),
            },
            indent=2,
        )
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, help="Miss detail JSONL")
    parser.add_argument("--data", action="append", required=True, help="dataset=feature_dir")
    parser.add_argument("--name", required=True, help="Output dataset name")
    parser.add_argument("--model-label", default="", help="Optional miss model_label filter")
    parser.add_argument("--min-ev-loss", type=float, default=0.0)
    parser.add_argument("--target-k", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
