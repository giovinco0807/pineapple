"""Build a 3-max T3-BTN teacher corpus across local cores.

    python scripts/build_three_max_t3_corpus.py --count 20000 --workers 14 \
        --output D:/ofc_data/three_max_t3_btn/corpus.jsonl

Work is split into contiguous seed shards, one worker per shard, each writing
its own file plus manifest; the shards are then concatenated in shard order so
the corpus is a deterministic function of (base_seed, count, workers, config).
Shard files are kept, not deleted: a resumed or extended run reuses whichever
shards already exist and match their manifest hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from ofc_regular.three_max.teacher import TeacherConfig, write_corpus  # noqa: E402


def _shard_bounds(count: int, workers: int) -> list[tuple[int, int]]:
    """Contiguous (offset, size) blocks; earlier shards absorb the remainder."""
    base, extra = divmod(count, workers)
    bounds = []
    offset = 0
    for index in range(workers):
        size = base + (1 if index < extra else 0)
        if size:
            bounds.append((offset, size))
            offset += size
    return bounds


def _run_shard(args: tuple) -> dict:
    path, base_seed, size, samples, holdout, policy_sims, root_policy = args
    config = TeacherConfig(
        samples=samples,
        holdout_samples=holdout,
        rollout_policy_sims=policy_sims,
        root_policy=root_policy,
    )
    path = Path(path)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if path.exists() and manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        matches = (
            existing.get("rows") == size
            and existing.get("base_seed") == base_seed
            and existing.get("config_fingerprint") == config.fingerprint()
            and existing.get("content_sha256")
            == hashlib.sha256(path.read_bytes()).hexdigest()
        )
        if matches:
            existing["reused"] = True
            return existing
    manifest = write_corpus(
        output=path, count=size, base_seed=base_seed, config=config
    )
    manifest["reused"] = False
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=8_100_000)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--holdout-samples", type=int, default=64)
    parser.add_argument("--rollout-policy-sims", type=int, default=4)
    parser.add_argument("--root-policy", choices=("hu", "mc"), default="hu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    shard_dir = args.output.parent / (args.output.stem + "_shards")
    shard_dir.mkdir(parents=True, exist_ok=True)
    bounds = _shard_bounds(args.count, args.workers)
    jobs = [
        (
            str(shard_dir / f"shard_{index:03d}.jsonl"),
            args.base_seed + offset,
            size,
            args.samples,
            args.holdout_samples,
            args.rollout_policy_sims,
            args.root_policy,
        )
        for index, (offset, size) in enumerate(bounds)
    ]

    print(
        f"{args.count} roots over {len(jobs)} shards "
        f"(samples={args.samples}, holdout={args.holdout_samples})",
        flush=True,
    )
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        manifests = list(pool.map(_run_shard, jobs))
    elapsed = time.time() - started

    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    rows = 0
    actions = 0
    with args.output.open("wb") as handle:
        for job, manifest in zip(jobs, manifests):
            payload = Path(job[0]).read_bytes()
            handle.write(payload)
            digest.update(payload)
            rows += manifest["rows"]
            actions += manifest["actions"]

    reused = sum(1 for manifest in manifests if manifest.get("reused"))
    summary = {
        "schema": "regular_ofc_3max_corpus_v1",
        "rows": rows,
        "actions": actions,
        "shards": len(jobs),
        "shards_reused": reused,
        "base_seed": args.base_seed,
        "seed_block": [args.base_seed, args.base_seed + args.count - 1],
        "config_fingerprint": manifests[0]["config_fingerprint"],
        "content_sha256": digest.hexdigest(),
        "wall_seconds": round(elapsed, 1),
        "core_seconds_per_root": round(
            sum(m["elapsed_seconds"] for m in manifests) / max(rows, 1), 4
        ),
        "shard_files": [Path(job[0]).name for job in jobs],
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
