"""Run one HU street-teacher shard and publish its labels as they appear.

A shard is a contiguous line range of a requests file that every worker
downloads whole: `(job, start, count)`.  Roots carry their line number as
their id, so a label object is named by an id that is unique across the whole
job and no worker needs to know what any other worker is doing.  Re-running a
shard unchanged is the recovery procedure for a preemption.

# Why a line range and not a seed

The T1-vs-FL fleet deals its roots from their seeds, so a shard's work is
implied by two numbers.  These roots cannot be: they are positions the
generation-3 chain actually reached, read off a two-seat trace.  The trace is
the input, so the input file ships and the shard names a slice of it.

# Publishing

One object per root, create-only, uploaded the moment the labeler flushes the
root's line -- the labeler is told `--batch-roots` small enough that lines
appear steadily.  Spot instances are preempted mid-shard; a shard that
publishes only at the end loses everything it did.

# Encoding

`--encode` has the worker turn its own labels into the trainer's feature
matrices before it exits.  At T0 that step costs as much as the labelling
did -- both are 232 joint blocks a root, four hundred sampled completions
each -- so leaving it at home would put a ten-hour job back on the
workstation to save twenty minutes of fleet time.  The shard's three splits
go up as `enc/{job}/{start}_{split}.npz`; the split assignment is a hash of
the hand, so shards partition the same way whether they are encoded together
or apart.

# Resume

Already-published ids are read from the object store, never from local disk
(a recreated Spot instance boots with an empty one), and are dropped from the
slice before the labeler starts, so a resumed shard pays for nothing twice.
A listing that cannot be read is fatal: labelling a shard whose state is
unknown is how a run silently doubles its bill.

Labelling and encoding are checked independently, because a worker preempted
between them comes back with its labels published and no features: the
resumed shard finds nothing to label, and would skip the encode too if the
two shared a condition.

Usage (on a worker; see startup_hu_street.sh):
    python3 -m ai.tutor.fleet.run_hu_street_shard --bucket B --prefix hu-street \\
        --run-id t0-r1 --job t0_btn --seat btn --start 0 --count 250 \\
        --requests t0_btn_requests.jsonl --value-model v1s.npz \\
        --work-dir /var/lib/w --workspace-root /var/lib/w/src
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from ai.tutor.fleet.gcs import GCLOUD, listing, say, upload


def published_ids(bucket: str, prefix: str, run_id: str, job: str) -> set[str]:
    """Root ids already in the bucket for this run's job."""
    url = f"gs://{bucket}/{prefix}/runs/{run_id}/labels/{job}/"
    return {
        name[: -len(".jsonl")]
        for name in listing(url)
        if name.endswith(".jsonl")
    }


class Publisher:
    """Ships whole lines out of the labeler's growing output file."""

    def __init__(self, path: Path, base: str) -> None:
        self.path = path
        self.base = base
        self.offset = 0
        self.tail = ""
        self.count = 0

    def tick(self) -> int:
        if not self.path.exists():
            return 0
        with self.path.open("r", encoding="utf-8") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
            self.offset = handle.tell()
        if not chunk:
            return 0
        self.tail += chunk
        *lines, self.tail = self.tail.split("\n")
        shipped = 0
        for line in lines:
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("actions"):
                raise RuntimeError(f"root {record['id']} came back with no actions")
            upload(line + "\n", f"{self.base}/{record['id']}.jsonl")
            self.count += 1
            shipped += 1
        return shipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--job", required=True,
                        help="names the label namespace, e.g. t0_btn")
    parser.add_argument("--seat", choices=["btn", "bb"], required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--value-model", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--batch-roots", type=int, default=8)
    parser.add_argument("--joint-samples", type=int, default=400)
    parser.add_argument("--publish-seconds", type=float, default=20.0)
    parser.add_argument("--extra-args", default="",
                        help="passed through to the teacher, e.g. "
                             "'--no-traced-opponent --opp-draws 4'")
    parser.add_argument("--encode", action="store_true",
                        help="also publish this shard's feature matrices")
    parser.add_argument("--labels", type=Path, default=None,
                        help="a finished labels file for the whole job.  With "
                             "it the shard skips the teacher and only encodes "
                             "-- the way a street whose labels already exist "
                             "gets its features off the workstation.")
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    base = f"gs://{args.bucket}/{args.prefix}/runs/{args.run_id}/labels/{args.job}"

    rows = [
        line for line in args.requests.open(encoding="utf-8") if line.strip()
    ][args.start:args.start + args.count]
    if not rows:
        raise SystemExit(f"FATAL: slice [{args.start}, +{args.count}) is empty")

    if args.labels is not None:
        # Encode-only: the labels are already made, and what is expensive is
        # what is left.  Matching on id rather than on line number, so this
        # cannot silently pair a request with another root's label.
        wanted = {json.loads(line)["id"] for line in rows}
        held = {}
        for line in args.labels.open(encoding="utf-8"):
            if line.strip():
                record = json.loads(line)
                if record["id"] in wanted:
                    held[record["id"]] = line
        absent = sorted(wanted - set(held))
        if absent:
            raise SystemExit(
                f"FATAL: {len(absent)} of the slice's roots are unlabelled, "
                f"first {absent[:5]}"
            )
        args.work_dir.mkdir(parents=True, exist_ok=True)
        (args.work_dir / "labels.jsonl").write_text(
            "".join(held[json.loads(line)["id"]] for line in rows),
            encoding="utf-8",
        )
        say(f"{args.job} [{args.start}, +{len(rows)}): encode only")
        encode_shard(args, rows, base)
        return
    done_ids = published_ids(args.bucket, args.prefix, args.run_id, args.job)
    remaining = [line for line in rows if json.loads(line)["id"] not in done_ids]
    say(
        f"{args.job} [{args.start}, +{len(rows)}): "
        f"{len(rows) - len(remaining)} already published, {len(remaining)} to do"
    )
    if not remaining:
        # The shard was preempted after publishing its labels and before
        # publishing its features.  Returning here -- which is what this did
        # -- makes the relaunch a no-op and leaves the features missing
        # forever, with the run looking complete because every label is there.
        say("all labels already published")
        if args.encode:
            encode_shard(args, rows, base)
        return

    slice_path = args.work_dir / "slice.jsonl"
    slice_path.write_text("".join(remaining), encoding="utf-8")
    out_path = args.work_dir / "labels.jsonl"
    if out_path.exists():
        out_path.unlink()

    command = [
        sys.executable, "-m", "ai.tutor.hu_street_teacher",
        "--requests", str(slice_path),
        "--seat", args.seat,
        "--value-model", str(args.value_model),
        "--out", str(out_path),
        "--batch-roots", str(args.batch_roots),
        "--joint-samples", str(args.joint_samples),
        "--workspace-root", str(args.workspace_root),
    ] + args.extra_args.split()
    say(" ".join(command))
    started = time.time()
    teacher = subprocess.Popen(command, cwd=str(args.workspace_root))
    publisher = Publisher(out_path, base)
    try:
        while True:
            code = teacher.poll()
            publisher.tick()
            if code is not None:
                break
            time.sleep(args.publish_seconds)
        publisher.tick()
    finally:
        if teacher.poll() is None:
            teacher.kill()
    elapsed = time.time() - started
    if teacher.returncode != 0:
        raise SystemExit(
            f"FATAL: labeler exited {teacher.returncode} after "
            f"{publisher.count}/{len(remaining)} roots"
        )
    if publisher.count != len(remaining):
        raise SystemExit(
            f"FATAL: labeler exited 0 with {publisher.count}/{len(remaining)} "
            "roots published"
        )
    say(
        f"published {publisher.count} roots in {elapsed:.0f}s "
        f"({elapsed / publisher.count:.1f} s/root)"
    )
    if args.encode:
        encode_shard(args, rows, base)


SPLITS = ("fit", "dev", "test")


def encode_shard(args, rows: list[str], label_base: str) -> None:
    """Turn this shard's labels into the trainer's matrices and publish them."""
    enc_base = (
        f"gs://{args.bucket}/{args.prefix}/runs/{args.run_id}/enc/{args.job}"
    )
    have = set(listing(enc_base + "/"))
    wanted = {f"{args.start:07d}_{split}.npz" for split in SPLITS}
    if wanted <= have:
        say("features already published")
        return

    # The encoder needs every label of the slice, and after a preemption some
    # of them are only in the bucket.
    out_path = args.work_dir / "labels.jsonl"
    local = {}
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            if line.strip():
                local[json.loads(line)["id"]] = line
    ids = [json.loads(line)["id"] for line in rows]
    absent = [root for root in ids if root not in local]
    if absent:
        say(f"fetching {len(absent)} labels published by an earlier attempt")
        fetched = args.work_dir / "fetched"
        fetched.mkdir(parents=True, exist_ok=True)
        done = subprocess.run(
            [GCLOUD, "storage", "cp"]
            + [f"{label_base}/{root}.jsonl" for root in absent]
            + [str(fetched)],
            capture_output=True, text=True,
        )
        if done.returncode != 0:
            raise SystemExit(f"FATAL: refetch failed: {done.stderr.strip()[:300]}")
        for root in absent:
            text = (fetched / f"{root}.jsonl").read_text(encoding="utf-8")
            local[root] = text.strip() + "\n"

    full_labels = args.work_dir / "labels_full.jsonl"
    full_labels.write_text("".join(local[root] for root in ids), encoding="utf-8")
    full_requests = args.work_dir / "slice_full.jsonl"
    full_requests.write_text("".join(rows), encoding="utf-8")
    enc_dir = args.work_dir / "enc"

    command = [
        sys.executable, "-m", "ai.tutor.encode_hu_teacher",
        "--labels", str(full_labels),
        "--requests", str(full_requests),
        "--out-dir", str(enc_dir),
        "--workspace-root", str(args.workspace_root),
    ]
    say(" ".join(command))
    started = time.time()
    done = subprocess.run(command, cwd=str(args.workspace_root))
    if done.returncode != 0:
        raise SystemExit(f"FATAL: encoder exited {done.returncode}")
    for split in SPLITS:
        path = enc_dir / f"{split}.npz"
        if not path.exists():
            raise SystemExit(f"FATAL: encoder wrote no {split}.npz")
        url = f"{enc_base}/{args.start:07d}_{split}.npz"
        pushed = subprocess.run(
            [GCLOUD, "storage", "cp", "--if-generation-match=0",
             str(path), url],
            capture_output=True, text=True,
        )
        blurb = (pushed.stderr or "") + (pushed.stdout or "")
        if pushed.returncode != 0 and "recondition" not in blurb and "412" not in blurb:
            raise SystemExit(f"FATAL: cannot publish {url}: {blurb.strip()[:300]}")
    say(f"features published in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
