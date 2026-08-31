"""Assemble a fleet job's published labels into the one file the trainer reads.

Workers publish one object per root, named by the root's id, so the job's
state lives in the listing and this reads it: what arrived, what did not, and
whether what arrived is intact.  The missing ids are printed rather than
silently skipped -- a teacher file that is short by two hundred roots trains
a model that looks fine and is quietly built on less than was paid for.

Output is ordered by id, which is the order the requests file has, so the
labels line up with `--requests` for every downstream encoder without any of
them having to sort.

Usage:
    python -m ai.tutor.fleet.collect_hu_labels --run-id t0-r1 --job t0_btn \\
        --expect 10000 --out D:/ofc_data/hu/t0_btn_labels.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from ai.tutor.fleet.gcs import GCLOUD, listing

BUCKET = "pokerhu-ofc-solver-485418-training"
PREFIX = "hu-street"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expect", type=int, default=0,
                        help="roots the job was launched with; 0 skips the check")
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument("--keep", type=Path, default=None,
                        help="download here instead of a temporary directory")
    args = parser.parse_args()

    base = f"gs://{args.bucket}/{args.prefix}/runs/{args.run_id}/labels/{args.job}"
    names = [n for n in listing(base + "/") if n.endswith(".jsonl")]
    print(f"{base}: {len(names)} objects")
    if args.expect:
        have = {int(n[: -len(".jsonl")]) for n in names}
        missing = sorted(set(range(args.expect)) - have)
        if missing:
            preview = ", ".join(str(m) for m in missing[:20])
            more = f" ... (+{len(missing) - 20})" if len(missing) > 20 else ""
            print(f"MISSING {len(missing)} of {args.expect}: {preview}{more}")
        else:
            print(f"complete: all {args.expect} roots present")

    with tempfile.TemporaryDirectory() as tmp:
        into = args.keep or Path(tmp)
        into.mkdir(parents=True, exist_ok=True)
        done = subprocess.run(
            [GCLOUD, "storage", "cp", f"{base}/*.jsonl", str(into)],
            capture_output=True, text=True,
        )
        if done.returncode != 0:
            raise SystemExit(f"FATAL: download failed: {done.stderr.strip()[:400]}")
        files = sorted(into.glob("*.jsonl"), key=lambda p: int(p.stem))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with args.out.open("w", encoding="utf-8", newline="\n") as out:
            for path in files:
                text = path.read_text(encoding="utf-8").strip()
                if not text:
                    raise SystemExit(f"FATAL: {path} is empty")
                record = json.loads(text)
                if record["id"] != path.stem:
                    raise SystemExit(
                        f"FATAL: {path} holds root {record['id']}"
                    )
                if not record.get("actions"):
                    raise SystemExit(f"FATAL: root {record['id']} has no actions")
                out.write(text + "\n")
                written += 1
    print(f"{args.out}: {written} roots")


if __name__ == "__main__":
    main()
