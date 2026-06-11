"""Build an on-disk feature cache for HU-aware Turn3 JSONL teachers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np

from .hu_turn3_model import HU_FEATURE_DIM, sample_to_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-samples", type=int, default=200)
    parser.add_argument("--max-outstanding", type=int)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _iter_jsonl_lines(path: Path, max_samples: int | None = None):
    seen = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            if max_samples is not None and seen >= max_samples:
                break
            yield line
            seen += 1


def _count_samples(path: Path, max_samples: int | None) -> dict[str, Any]:
    action_counts: list[int] = []
    source_counts: Counter[str] = Counter()
    started_at = time.time()
    for line in _iter_jsonl_lines(path, max_samples):
        sample = json.loads(line)
        actions = sample.get("actions", ())
        if not actions:
            raise ValueError("HU teacher sample has no actions")
        action_counts.append(len(actions))
        source_counts[str(sample.get("source", "unknown"))] += 1
    offsets = np.zeros(len(action_counts) + 1, dtype=np.int64)
    if action_counts:
        offsets[1:] = np.cumsum(np.asarray(action_counts, dtype=np.int64))
    return {
        "action_counts": np.asarray(action_counts, dtype=np.int16),
        "offsets": offsets,
        "source_counts": dict(source_counts),
        "elapsed_seconds": time.time() - started_at,
    }


def _encode_chunk(start_sample: int, lines: list[str], dtype_name: str) -> dict[str, Any]:
    feature_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    action_counts: list[int] = []
    source_counts: Counter[str] = Counter()
    for line in lines:
        sample = json.loads(line)
        features, targets = sample_to_matrix(sample)
        feature_blocks.append(features.astype(dtype_name, copy=False))
        target_blocks.append(targets.astype(np.float32, copy=False))
        action_counts.append(int(targets.shape[0]))
        source_counts[str(sample.get("source", "unknown"))] += 1
    if feature_blocks:
        features_out = np.vstack(feature_blocks).astype(dtype_name, copy=False)
        targets_out = np.concatenate(target_blocks).astype(np.float32, copy=False)
    else:
        features_out = np.zeros((0, HU_FEATURE_DIM), dtype=dtype_name)
        targets_out = np.zeros(0, dtype=np.float32)
    return {
        "start_sample": start_sample,
        "sample_count": len(lines),
        "action_count": int(targets_out.shape[0]),
        "features": features_out,
        "targets": targets_out,
        "action_counts": np.asarray(action_counts, dtype=np.int16),
        "source_counts": dict(source_counts),
    }


def _submit_chunk(
    executor: ProcessPoolExecutor,
    start_sample: int,
    lines: list[str],
    dtype_name: str,
):
    return executor.submit(_encode_chunk, start_sample, lines, dtype_name)


def main() -> None:
    args = parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.chunk_samples <= 0:
        raise SystemExit("--chunk-samples must be positive")

    cache_dir = args.cache_dir.resolve()
    if cache_dir.exists() and any(cache_dir.iterdir()) and not args.force:
        raise SystemExit(f"cache dir is not empty: {cache_dir} (use --force)")
    cache_dir.mkdir(parents=True, exist_ok=True)

    dtype = np.dtype(args.dtype)
    started_at = time.time()
    count_info = _count_samples(args.input, args.max_samples)
    action_counts = count_info["action_counts"]
    offsets = count_info["offsets"]
    sample_count = int(action_counts.shape[0])
    action_count = int(offsets[-1])
    if sample_count <= 0 or action_count <= 0:
        raise SystemExit("no HU teacher samples")

    np.save(cache_dir / "sample_offsets.npy", offsets)
    np.save(cache_dir / "sample_action_counts.npy", action_counts)

    features_path = cache_dir / f"features.{args.dtype}.mmap"
    targets_path = cache_dir / "targets.float32.mmap"
    features = np.memmap(features_path, dtype=dtype, mode="w+", shape=(action_count, HU_FEATURE_DIM))
    targets = np.memmap(targets_path, dtype=np.float32, mode="w+", shape=(action_count,))

    max_outstanding = args.max_outstanding or max(args.workers * 2, 1)
    executor = ProcessPoolExecutor(max_workers=args.workers)
    futures = set()
    submitted_samples = 0
    written_samples = 0
    written_actions = 0
    combined_sources: Counter[str] = Counter()

    def consume(done_futures) -> None:
        nonlocal written_samples, written_actions
        for future in done_futures:
            result = future.result()
            start_sample = int(result["start_sample"])
            sample_total = int(result["sample_count"])
            action_start = int(offsets[start_sample])
            action_end = int(offsets[start_sample + sample_total])
            expected_actions = action_end - action_start
            if expected_actions != int(result["action_count"]):
                raise ValueError(
                    f"chunk action mismatch at sample {start_sample}: "
                    f"expected {expected_actions}, got {result['action_count']}"
                )
            features[action_start:action_end] = result["features"]
            targets[action_start:action_end] = result["targets"]
            written_samples += sample_total
            written_actions += expected_actions
            combined_sources.update(result["source_counts"])
            if args.progress_every > 0 and written_samples % args.progress_every < sample_total:
                print(
                    json.dumps(
                        {
                            "event": "cache_progress",
                            "samples": written_samples,
                            "actions": written_actions,
                            "elapsed_seconds": time.time() - started_at,
                        },
                        separators=(",", ":"),
                    ),
                    file=sys.stderr,
                    flush=True,
                )

    try:
        chunk_lines: list[str] = []
        chunk_start = 0
        for line in _iter_jsonl_lines(args.input, args.max_samples):
            if not chunk_lines:
                chunk_start = submitted_samples
            chunk_lines.append(line)
            submitted_samples += 1
            if len(chunk_lines) >= args.chunk_samples:
                futures.add(_submit_chunk(executor, chunk_start, chunk_lines, args.dtype))
                chunk_lines = []
            if len(futures) >= max_outstanding:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                consume(done)
        if chunk_lines:
            futures.add(_submit_chunk(executor, chunk_start, chunk_lines, args.dtype))
        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            consume(done)
    finally:
        executor.shutdown(wait=True, cancel_futures=False)

    features.flush()
    targets.flush()
    metadata = {
        "input": str(args.input),
        "cache_dir": str(cache_dir),
        "features_path": str(features_path),
        "targets_path": str(targets_path),
        "feature_dtype": args.dtype,
        "target_dtype": "float32",
        "feature_dim": HU_FEATURE_DIM,
        "samples": sample_count,
        "actions": action_count,
        "source_counts": dict(combined_sources) or count_info["source_counts"],
        "count_elapsed_seconds": count_info["elapsed_seconds"],
        "elapsed_seconds": time.time() - started_at,
        "workers": args.workers,
        "chunk_samples": args.chunk_samples,
    }
    (cache_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
