"""Build the sharpened dev set in resumable shards.

teach-t2 opens its output with File::create, so one monolithic 4-hour run
loses everything to one kill.  This driver shards the 1,022 dev roots into
40-root pieces, skips any shard whose output already holds the right number
of lines, and merges at the end -- rerunning after a kill resumes.

Each pass uses its own stream offset (independent T3 draws); pass the pass
name and offset on the command line so the two can run sequentially or the
driver can be relaunched for either.

Usage:
    python run_sharpdev.py pass1 940000001
    python run_sharpdev.py pass2 950000001
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

SCRATCH = Path(r"C:\Users\Owner\AppData\Local\Temp\claude\C--Users-Owner--gemini-antigravity-scratch-ofc-pineapple\fbbaae4a-e47e-44b7-ab1b-cde058d5523e\scratchpad")
EXE = Path("C:/Users/Owner/.gemini/antigravity/scratch/ofc-pineapple/ai/rust_solver/target/release/fl_solver.exe")
WS = SCRATCH / "t2sharpen_ws"  # holds the fl_ev_prev config the labels were priced under
ROOTS = SCRATCH / "dev_roots_all.jsonl"
POOL = "D:/ofc_data/fl_pools/fl14_v1.jfl1"
SHARD = 40
T3_DRAWS = 96


def main() -> None:
    name, offset = sys.argv[1], int(sys.argv[2])
    lines = [l for l in ROOTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    # A four-hour asset outlives this session, so it lives on D:, not in the
    # session scratchpad.
    out_dir = Path("D:/ofc_data/lap4_t2_own/sharpdev") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for base in range(0, len(lines), SHARD):
        chunk = lines[base:base + SHARD]
        shard_dir = out_dir / f"shard_{base:05d}"
        done_file = shard_dir / "t2_labels.jsonl"
        if done_file.exists():
            have = sum(1 for l in done_file.read_text(encoding="utf-8").splitlines()
                       if l.strip())
            if have == len(chunk):
                continue
        shard_dir.mkdir(exist_ok=True)
        roots_path = shard_dir / "roots.jsonl"
        roots_path.write_text("\n".join(chunk) + "\n", encoding="utf-8")
        result = subprocess.run(
            [str(EXE), "teach-t2", "--roots-file", str(roots_path),
             "--out-dir", str(shard_dir), "--pool", POOL,
             "--opponents", "60", "--own-only",
             "--t3-draws", str(T3_DRAWS), "--t4-draws", "0",
             "--stream-offset", str(offset),
             "--root-offset", str(base)],
            cwd=str(WS), capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(f"shard {base} failed:\n{result.stderr[-800:]}")
        done = base + len(chunk)
        rate = done / max(time.time() - started, 1e-9)
        print(f"[{name}] {done}/{len(lines)} roots, "
              f"eta {(len(lines) - done) / max(rate, 1e-9) / 60:.0f} min", flush=True)
    merged = out_dir / "t2_labels.jsonl"
    with merged.open("w", encoding="utf-8") as out:
        for base in range(0, len(lines), SHARD):
            text = (out_dir / f"shard_{base:05d}" / "t2_labels.jsonl").read_text(encoding="utf-8")
            out.write(text if text.endswith("\n") else text + "\n")
    total = sum(1 for l in merged.read_text(encoding="utf-8").splitlines() if l.strip())
    print(f"[{name}] merged {total} roots -> {merged}", flush=True)


if __name__ == "__main__":
    main()
