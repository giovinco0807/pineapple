"""Run one T1-vs-FL label shard and publish its labels as they are produced.

The shard is `(seed, roots)`; the roots are dealt deterministically from those
two numbers (`t1_root_requests`), so this needs no coordination with any other
worker and a preempted shard is re-run by relaunching it unchanged.

# Publishing

One object per root, named by its absolute seed, uploaded create-only the
moment the labeler flushes that root's line -- not collected and shipped when
the shard finishes.  Spot instances are preempted mid-run: a shard that
publishes only at the end loses everything, which has cost this project an
hour of work twice.  `--chunk-size 1` on the labeler is what makes a root's
line appear as soon as it exists.

Create-only (`--if-generation-match=0`) is what makes a re-run safe.  A root's
label is a function of its seed and the artefacts, so an object that already
exists is the same object; the precondition failure is the correct no-op,
not a conflict.

# Resume

What this shard has already published is read from the object store, never
from local disk -- a recreated Spot instance boots with an empty one.  A
listing that cannot be read is fatal: proceeding would mean re-labelling a
shard whose state is unknown.

Usage (on a worker; see startup_t1_vs_fl.sh):
    python -m ai.tutor.fleet.run_t1_shard --bucket B --prefix fl14-t1 \
        --run-id t1-r1 --seed 101000000 --roots 20 --work-dir /var/lib/w \
        --solver ./t4_first_exact --pool pool.jfl1 \
        --t2-model t2.bin --t3-model t3.bin --fl-ev-config ai/config/fl_ev.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ai.tutor.fleet.t1_root_requests import add_arguments, requests_for, write_requests

# On Windows the entry point is gcloud.cmd, which subprocess cannot resolve
# from the bare name without a shell.
GCLOUD = shutil.which("gcloud") or "gcloud"
NO_OBJECTS = "matched no objects"


def say(message: str) -> None:
    print(f"[shard] {message}", flush=True)


def published_ids(bucket: str, prefix: str, run_id: str) -> set[str]:
    """Root ids already in the bucket for this run."""
    url = f"gs://{bucket}/{prefix}/runs/{run_id}/labels/"
    done = subprocess.run(
        [GCLOUD, "storage", "ls", url], capture_output=True, text=True
    )
    if done.returncode != 0:
        if NO_OBJECTS in (done.stderr or ""):
            return set()
        raise SystemExit(f"FATAL: cannot read {url}: {done.stderr.strip()[:400]}")
    out = set()
    for line in done.stdout.splitlines():
        name = line.rsplit("/", 1)[-1].strip()
        if name.endswith(".jsonl"):
            out.add(name[: -len(".jsonl")])
    return out


def upload_once(text: str, url: str, *, create_only: bool = True) -> bool:
    """Write `text` to `url`.  False means "already there", which is benign."""
    command = [GCLOUD, "storage", "cp"]
    if create_only:
        command += ["--if-generation-match=0"]
    command += ["-", url]
    done = subprocess.run(command, input=text, capture_output=True, text=True)
    if done.returncode == 0:
        return True
    blurb = (done.stderr or "") + (done.stdout or "")
    if "recondition" in blurb or "412" in blurb:
        return False
    raise RuntimeError(f"upload {url} failed: {blurb.strip()[:300]}")


def upload(text: str, url: str, *, attempts: int = 3) -> bool:
    """`upload_once` with a short retry: a blip on a startup upload would
    otherwise cost the whole instance, and one on a label upload would cost a
    root that has already been paid for."""
    for attempt in range(attempts):
        try:
            return upload_once(text, url)
        except RuntimeError as error:
            if attempt == attempts - 1:
                raise
            say(f"retrying upload of {url}: {error}")
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


class Publisher:
    """Ships whole lines out of the labeler's growing output file."""

    def __init__(self, path: Path, bucket: str, prefix: str, run_id: str) -> None:
        self.path = path
        self.base = f"gs://{bucket}/{prefix}/runs/{run_id}/labels"
        self.offset = 0
        self.tail = ""
        self.count = 0
        self.first_at: float | None = None
        self.last_at: float | None = None

    def tick(self) -> int:
        """Publish every complete line written since the last call."""
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
            root_id = record["id"]
            if not record.get("actions"):
                raise RuntimeError(f"root {root_id} came back with no actions")
            upload(line + "\n", f"{self.base}/{root_id}.jsonl")
            now = time.time()
            self.first_at = self.first_at if self.first_at is not None else now
            self.last_at = now
            self.count += 1
            shipped += 1
        if shipped:
            say(f"published {shipped} label(s), {self.count} this attempt")
        return shipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default="fl14-t1")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--roots", type=int, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--t2-model", type=Path, required=True)
    parser.add_argument("--t3-model", type=Path, required=True)
    parser.add_argument("--fl-ev-config", type=Path, required=True)
    parser.add_argument("--publish-seconds", type=int, default=30)
    add_arguments(parser)
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    knobs = dict(
        opp_count=args.opp_count,
        t2_samples=args.t2_samples,
        t3_samples=args.t3_samples,
        t4_draw_sample=args.t4_draw_sample,
        pool_opponents=args.pool_opponents,
        truncate_depth=None if args.truncate_depth < 0 else args.truncate_depth,
    )

    # Provenance: the shard's full request set, which is the same file on every
    # attempt, so create-only keeps the first and later attempts no-op.
    full = args.work_dir / "requests_full.jsonl"
    write_requests(full, requests_for(seed=args.seed, roots=args.roots, **knobs))
    upload(
        full.read_text(encoding="utf-8"),
        f"gs://{args.bucket}/{args.prefix}/runs/{args.run_id}"
        f"/requests/{args.seed}.jsonl",
    )

    done_already = published_ids(args.bucket, args.prefix, args.run_id)
    mine = {str(args.seed + index) for index in range(args.roots)}
    todo = sorted(mine - done_already, key=int)
    say(f"{len(mine) - len(todo)} of {len(mine)} root(s) already published")
    if not todo:
        say("nothing left to do")
    else:
        request_path = args.work_dir / "in.jsonl"
        write_requests(
            request_path,
            requests_for(
                seed=args.seed, roots=args.roots, only_ids=set(todo), **knobs
            ),
        )
        out_path = args.work_dir / "out.jsonl"
        if out_path.exists():
            out_path.unlink()
        command = [
            str(args.solver),
            "--t1-vs-fl-library",
            "--fl-pool", str(args.pool),
            "--t1-t2-model", str(args.t2_model),
            "--t2-t3-model", str(args.t3_model),
            "--fl-ev-config", str(args.fl_ev_config),
            "--input", str(request_path),
            "--output", str(out_path),
            # One root per flush: the publisher can only ship what the labeler
            # has written, and a large chunk would hold whole batches of
            # finished work inside a process a preemption is about to kill.
            "--chunk-size", "1",
        ]
        say(" ".join(command))
        solver = subprocess.Popen(command)
        publisher = Publisher(out_path, args.bucket, args.prefix, args.run_id)
        while solver.poll() is None:
            time.sleep(args.publish_seconds)
            publisher.tick()
        publisher.tick()
        status = solver.returncode
        elapsed = time.time() - started
        rate = elapsed / max(publisher.count, 1)
        say(
            f"labeler exited {status}; {publisher.count} root(s) in "
            f"{elapsed:.0f}s ({rate:.1f}s/root wall)"
        )
        if publisher.count:
            span = (publisher.last_at or 0) - (publisher.first_at or 0)
            steady = span / max(publisher.count - 1, 1)
            upload(
                json.dumps(
                    {
                        "seed": args.seed,
                        "roots_requested": len(todo),
                        "roots_labelled": publisher.count,
                        "elapsed_seconds": round(elapsed, 1),
                        # First-root time includes the pool load; the steady
                        # rate is what a long shard actually costs.
                        "seconds_per_root_including_setup": round(rate, 2),
                        "seconds_per_root_steady": round(steady, 2),
                        "knobs": knobs,
                        "exit": status,
                    },
                    indent=2,
                ),
                f"gs://{args.bucket}/{args.prefix}/runs/{args.run_id}"
                f"/stats/{args.seed}_{int(started)}.json",
            )
        if status != 0:
            raise SystemExit(
                f"FATAL: labeler exited {status} "
                f"({publisher.count} published label(s) are still valid)"
            )

    remaining = mine - published_ids(args.bucket, args.prefix, args.run_id)
    if remaining:
        raise SystemExit(f"FATAL: {len(remaining)} root(s) never published")
    upload(
        f"done {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n",
        f"gs://{args.bucket}/{args.prefix}/runs/{args.run_id}/done/{args.seed}.txt",
    )
    say(f"SHARD_DONE {args.seed}")


if __name__ == "__main__":
    sys.exit(main())
