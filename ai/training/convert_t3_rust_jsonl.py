"""
Convert Rust T3 teacher JSONL rows into NPZ chunks consumed by train_t3_oracle_v2.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def flush_chunk(rows, output_dir: Path, chunk_idx: int):
    if not rows:
        return 0

    states = np.asarray([r["state"] for r in rows], dtype=np.float16)
    valid_masks = np.asarray(
        [r.get("valid_masks", r.get("action_masks")) for r in rows],
        dtype=np.bool_,
    )
    action_evs32 = np.asarray([r["action_evs"] for r in rows], dtype=np.float32)
    action_evs32[~valid_masks] = -1.0e4
    action_evs = action_evs32.astype(np.float16)
    actions = np.asarray([r["actions"] for r in rows], dtype=np.int64)
    rewards = np.asarray([r.get("rewards", r.get("best_evs", 0.0)) for r in rows], dtype=np.float32)
    turns = np.asarray([r.get("turns", 3) for r in rows], dtype=np.uint8)
    is_btn = np.asarray([r.get("is_btn", False) for r in rows], dtype=np.bool_)

    out = output_dir / f"t3_rust_{chunk_idx:06d}.npz"
    np.savez(
        out,
        states=states,
        action_evs=action_evs,
        valid_masks=valid_masks,
        actions=actions,
        rewards=rewards,
        turns=turns,
        is_btn=is_btn,
    )

    valid_counts = valid_masks.sum(axis=1)
    print(
        f"wrote {out} rows={len(rows)} "
        f"state_dim={states.shape[1]} valid_avg={valid_counts.mean():.1f} "
        f"ev=[{rewards.min():.2f},{rewards.max():.2f}]"
    )
    return len(rows)


def iter_input_files(input_patterns):
    files = []
    for pattern in input_patterns:
        matched = sorted(Path().glob(pattern)) if any(ch in pattern for ch in "*?[]") else [Path(pattern)]
        files.extend(matched)
    seen = set()
    for path in files:
        path = path.resolve()
        if path not in seen:
            seen.add(path)
            yield path


def main():
    parser = argparse.ArgumentParser(description="Convert Rust T3 JSONL to NPZ chunks")
    parser.add_argument("--input", nargs="+", required=True, help="Input JSONL file(s) or glob(s)")
    parser.add_argument("--output-dir", required=True, help="Output directory for NPZ chunks")
    parser.add_argument("--chunk-size", type=int, default=10000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    chunk_idx = 0
    total = 0
    for path in iter_input_files(args.input):
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if len(row["state"]) != 522:
                    raise ValueError(f"{path}:{line_no} state dim {len(row['state'])}, expected 522")
                if len(row["action_evs"]) != 27:
                    raise ValueError(f"{path}:{line_no} action_evs dim {len(row['action_evs'])}, expected 27")
                mask = row.get("valid_masks", row.get("action_masks"))
                if mask is None or len(mask) != 27:
                    raise ValueError(f"{path}:{line_no} missing/invalid mask")
                rows.append(row)
                if len(rows) >= args.chunk_size:
                    total += flush_chunk(rows, output_dir, chunk_idx)
                    rows.clear()
                    chunk_idx += 1

    if rows:
        total += flush_chunk(rows, output_dir, chunk_idx)

    print(f"done total_rows={total} chunks={chunk_idx + (1 if rows else 0)} output={output_dir}")


if __name__ == "__main__":
    main()
