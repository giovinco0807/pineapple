"""Launch the T1 own-label fleet (sharpening or mass production).

Sibling of `launch_t2_labelgen_fleet`: each instance prices a contiguous
line-window [start, start+count) of a requests object with
`t4_first_exact --t1-vs-fl-library`.  The request lines carry the sampling
knobs, so one launcher serves both the dev sharpening (ids prefixed p1/p2,
t2_samples raised) and any future mass relabel -- only the requests object
changes.
"""
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path

from ai.tutor.fleet.launch_hu_street_fleet import (
    GCLOUD, BUCKET, PREFIX, placement_plan, disk_type,
)

STARTUP = Path(__file__).resolve().parent / "startup_t1_labelgen.sh"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--roots", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--binstamp", required=True)
    parser.add_argument("--requests-object", required=True)
    parser.add_argument("--pool-object", default="fl14_v1.jfl1")
    parser.add_argument("--movers-object", default="t1_movers.tar.gz")
    parser.add_argument("--chunk", type=int, default=25)
    parser.add_argument("--machine-type", default="n2-highcpu-32")
    parser.add_argument("--reserve-cores", type=int, default=16)
    parser.add_argument("--placement-offset", type=int, default=0)
    parser.add_argument("--watchdog-seconds", type=int, default=14400)
    parser.add_argument("--skip-done", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    per_shard = -(-args.roots // args.shards)
    zones = placement_plan(
        args.shards + args.placement_offset, args.machine_type, args.reserve_cores
    )[args.placement_offset:]
    if len(zones) < args.shards:
        raise SystemExit(f"quota fits {len(zones)} instances, {args.shards} asked for")

    alive: set[str] = set()
    if args.skip_done:
        done = subprocess.run(
            [GCLOUD, "compute", "instances", "list",
             "--filter", f"name~^t1l-{args.run_id}-", "--format", "value(name)"],
            capture_output=True, text=True)
        alive = {l.strip() for l in done.stdout.splitlines() if l.strip()}

    made = 0
    for index in range(args.shards):
        start = args.start + index * per_shard
        count = min(per_shard, args.start + args.roots - start)
        if count <= 0:
            break
        name = f"t1l-{args.run_id}-{start:08d}"
        if name in alive:
            continue
        zone = zones[index]
        metadata = ",".join([
            f"t1l-bucket={BUCKET}", f"t1l-prefix={PREFIX}",
            f"t1l-run-id={args.run_id}", f"t1l-start={start}", f"t1l-count={count}",
            f"t1l-binstamp={args.binstamp}",
            f"t1l-pool-object={args.pool_object}",
            f"t1l-requests-object={args.requests_object}",
            f"t1l-movers-object={args.movers_object}",
            f"t1l-chunk={args.chunk}",
            f"t1l-watchdog-seconds={args.watchdog_seconds}",
        ])
        if args.dry_run:
            print(f"would launch {name} {zone} [{start}, +{count})")
            made += 1
            continue
        done = subprocess.run(
            [GCLOUD, "compute", "instances", "create", name,
             "--zone", zone, "--machine-type", args.machine_type,
             "--provisioning-model", "SPOT",
             "--instance-termination-action", "DELETE",
             "--scopes", "cloud-platform",
             "--image-family", "debian-12", "--image-project", "debian-cloud",
             "--boot-disk-type", disk_type(args.machine_type),
             "--boot-disk-size", "30GB",
             "--metadata-from-file", f"startup-script={STARTUP}",
             "--metadata", metadata],
            capture_output=True, text=True)
        if done.returncode == 0:
            made += 1
            if made == 1 or made == args.shards:
                print(f"launched {name} {zone} [{start}, +{count})")
        else:
            print(f"FAILED {name} {zone}: "
                  f"{done.stderr.strip().splitlines()[-1] if done.stderr else '?'}")
    print(f"{made} instances, {per_shard} request-lines each")


if __name__ == "__main__":
    main()
