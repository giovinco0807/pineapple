"""Watch a fleet run: delete what finished, and name what went missing.

Two jobs, and the second is the reason this runs unattended rather than being
checked now and then.

# Deleting

The worker's own `gcloud compute instances delete` is refused: the label
service account holds storage rights but not `compute.instances.delete`, so
the last line of every startup script is a no-op and a finished worker keeps
running -- and keeps billing -- until `--max-run-duration` fires.  That is
the shape of the incident this project already paid for once, with 84
stopped-but-undeleted instances.  Rather than widen the worker's permissions,
the deletion happens here, under the operator's own credentials, driven by
the `generate_ok` marker a worker publishes when its labeler returns cleanly.

`--max-run-duration` with `--instance-termination-action DELETE` stays
underneath as the backstop: GCE enforces it, not the worker, so a hung worker
-- or a watcher that is not running -- still cannot bill forever.

# Reporting what is missing

A Spot worker that is preempted simply stops existing.  It publishes no
marker and prints no error, and the fleet looks smaller every time you count
it; the labels it did not write are noticed much later, by whatever trains on
a file that is quietly short.  So this tracks the shard starts it expects
(`--shards`), and calls a shard MISSING once its instance is gone without a
marker -- with the exact `--skip-done` relaunch that repairs it.

It also watches the counts move.  A fleet where every instance is alive and
no object has appeared for ten minutes is not working slowly, it is stuck,
and the difference is only visible over time.
"""
from __future__ import annotations

import argparse
import subprocess
import time

from ai.tutor.fleet.gcs import GCLOUD, listing

BUCKET = "pokerhu-ofc-solver-485418-training"
PREFIX = "hu-street"


def instances(run_id: str) -> dict[str, str]:
    """Live worker instances of this run, name -> zone."""
    done = subprocess.run(
        [GCLOUD, "compute", "instances", "list",
         "--filter", f"name~^hu-{run_id}-",
         "--format", "csv[no-heading](name,zone)"],
        capture_output=True, text=True,
    )
    if done.returncode != 0:
        raise SystemExit(f"FATAL: cannot list instances: {done.stderr.strip()[:300]}")
    out = {}
    for line in done.stdout.splitlines():
        if "," in line:
            name, zone = line.strip().split(",", 1)
            out[name] = zone.rsplit("/", 1)[-1]
    return out


def markers(bucket: str, prefix: str, run_id: str) -> set[tuple[str, int]]:
    """(job, start) pairs whose worker published a clean-exit marker."""
    found = set()
    for marker in listing(f"gs://{bucket}/{prefix}/runs/{run_id}/progress/"):
        if not marker.endswith("_generate_ok.txt"):
            continue
        job, _, start = marker[: -len("_generate_ok.txt")].rpartition("_")
        if start.isdigit():
            found.add((job, int(start)))
    return found


def instance_name(run_id: str, job: str, start: int) -> str:
    return f"hu-{run_id}-{job.replace('_', '-')}-{start:07d}"


def counts(bucket: str, prefix: str, run_id: str, job: str) -> tuple[int, int]:
    labels = len(listing(f"gs://{bucket}/{prefix}/runs/{run_id}/labels/{job}/"))
    features = len(listing(f"gs://{bucket}/{prefix}/runs/{run_id}/enc/{job}/"))
    return labels, features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--jobs", default="",
                        help="comma-separated jobs in this run")
    parser.add_argument("--roots", type=int, default=0,
                        help="roots per job, for the progress line")
    parser.add_argument("--shards", type=int, default=0,
                        help="shards launched per job.  Without it a preempted "
                             "shard cannot be told from one that never existed.")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--stall-minutes", type=float, default=45.0,
                        help="quiet time before a live fleet is called stuck.  "
                             "Must exceed one encode pass: that phase writes "
                             "nothing until it finishes, so a shorter window "
                             "reports every healthy run as stalled.")
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    jobs = [j for j in args.jobs.split(",") if j]
    per_shard = -(-args.roots // args.shards) if (args.roots and args.shards) else 0
    expected = {
        (job, index * per_shard)
        for job in jobs
        for index in range(args.shards)
    } if args.shards else set()

    started = time.time()
    last_total = -1
    last_move = time.time()
    reported_missing: set[tuple[str, int]] = set()

    while True:
        live = instances(args.run_id)
        # Markers accumulate across every job of the run; counting another
        # job's against this one's shard budget reads as progress that is not
        # there.
        done = {
            shard for shard in markers(args.bucket, args.prefix, args.run_id)
            if not jobs or shard[0] in jobs
        }
        for job, start in sorted(done):
            name = instance_name(args.run_id, job, start)
            if name not in live:
                continue
            killed = subprocess.run(
                [GCLOUD, "compute", "instances", "delete", name,
                 "--zone", live[name], "--quiet"],
                capture_output=True, text=True,
            )
            blurb = killed.stderr or ""
            if killed.returncode == 0:
                verdict = "deleted"
            elif "was not found" in blurb or "Could not fetch resource" in blurb:
                # Another pass, or the instance's own max-run-duration, got
                # there first.  Gone is the outcome this asked for.
                verdict = "already gone"
            else:
                verdict = f"FAILED ({blurb.strip()[:120]})"
            print(f"  {name}: {verdict}", flush=True)
            live.pop(name, None)

        # A shard with neither an instance nor a marker was preempted.  Say so
        # once, with the command that fixes it.
        missing = sorted(
            shard for shard in expected
            if shard not in done
            and instance_name(args.run_id, *shard) not in live
        )
        for shard in missing:
            if shard in reported_missing:
                continue
            reported_missing.add(shard)
            print(
                f"  MISSING {instance_name(args.run_id, *shard)} "
                f"(preempted before publishing) -- relaunch the job with "
                f"--skip-done", flush=True,
            )

        totals = {job: counts(args.bucket, args.prefix, args.run_id, job)
                  for job in jobs}
        total = sum(labels + features for labels, features in totals.values())
        if total != last_total:
            last_total, last_move = total, time.time()
        idle = (time.time() - last_move) / 60.0

        detail = " ".join(
            f"{job}={labels}L/{features}F" for job, (labels, features) in totals.items()
        )
        share = f"/{args.roots * len(jobs)}" if args.roots else ""
        print(
            f"[{time.time() - started:6.0f}s] {len(live)} live, "
            f"{len(done)}/{len(expected) or '?'} done, "
            f"{len(missing)} missing, objects {total}{share} {detail}",
            flush=True,
        )
        if live and idle >= args.stall_minutes:
            print(
                f"  STALLED: {len(live)} instances alive and nothing published "
                f"for {idle:.0f} minutes", flush=True,
            )

        if args.once:
            return
        if not live and (not expected or done | set(missing) >= expected):
            print(
                f"fleet drained: {len(done)} shards finished, "
                f"{len(missing)} lost to preemption", flush=True,
            )
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
