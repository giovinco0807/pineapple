"""Encode a 3-max T3 corpus once, in parallel, into an .npz the trainer reads.

    python scripts/encode_three_max_t3.py --corpus corpus.jsonl \
        --output corpus.features.npz --workers 14

Encoding is ~5 ms per action and a corpus has hundreds of thousands of them, so
doing it inside the training loop would cost more than the training.  The cache
records the corpus hash and the feature schema: a trainer that reads a cache
built from different bytes, or from a different encoder, must not silently use
it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from ofc_regular.three_max.features import (  # noqa: E402
    FEATURE_SCHEMA,
    FEATURE_SIZE,
    encode_record_action,
)


def _encode_chunk(lines: list[str]) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    features: list[list[float]] = []
    labels: list[float] = []
    widths: list[int] = []
    seeds: list[int] = []
    for line in lines:
        record = json.loads(line)
        for index in range(len(record["actions"])):
            features.append(list(encode_record_action(record, index).features))
            labels.append(record["actions"][index]["ev"])
        widths.append(len(record["actions"]))
        seeds.append(record["seed"])
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        widths,
        seeds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=200)
    args = parser.parse_args()

    lines = args.corpus.read_text(encoding="utf-8").splitlines()
    chunks = [lines[i : i + args.chunk] for i in range(0, len(lines), args.chunk)]
    print(f"{len(lines)} roots in {len(chunks)} chunks", flush=True)

    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_encode_chunk, chunks))

    features = np.concatenate([r[0] for r in results])
    labels = np.concatenate([r[1] for r in results])
    widths = np.asarray([w for r in results for w in r[2]], dtype=np.int32)
    seeds = np.asarray([s for r in results for s in r[3]], dtype=np.int64)
    assert features.shape == (len(labels), FEATURE_SIZE)
    assert widths.sum() == len(labels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features,
        labels=labels,
        widths=widths,
        seeds=seeds,
        meta=np.asarray(
            [
                json.dumps(
                    {
                        "feature_schema": FEATURE_SCHEMA,
                        "feature_size": FEATURE_SIZE,
                        "corpus": str(args.corpus),
                        "corpus_sha256": hashlib.sha256(
                            args.corpus.read_bytes()
                        ).hexdigest(),
                        "roots": int(len(widths)),
                        "actions": int(len(labels)),
                    },
                    sort_keys=True,
                )
            ]
        ),
    )
    elapsed = time.time() - started
    print(
        f"encoded {len(labels)} actions from {len(widths)} roots in {elapsed:.1f}s "
        f"({elapsed / max(len(labels), 1) * 1000:.2f} ms/action wall)",
        flush=True,
    )


if __name__ == "__main__":
    main()
