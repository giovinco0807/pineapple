"""Launch the T2 own-label production fleet.

Thin sibling of `launch_ship_gate_fleet` -- same placement plan and quota
behaviour; different startup script and metadata.  Each instance deals and
prices a contiguous root window [start, start+count) of the canonical T2 deal
stream (seed 3238398113, stream-offset 0), continuing exactly where
`t2_labels_own_10k.jsonl` stopped, so the corpus stays one stream and ids
never collide.

Recipe defaults are the sharpened-corpus recipe measured on 2026-09-01:
96 T3 draws with the pilot-32/keep-2 shortcut (within-root differential
<= 0.016, 4x cheaper than pricing every T3 placement exactly).
"""
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path

from ai.tutor.fleet.launch_hu_street_fleet import (
    GCLOUD, BUCKET, PREFIX, placement_plan, disk_type,
)

STARTUP = Path(__file__).resolve().parent / "startup_t2_labelgen.sh"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--start", type=int, required=True,
                        help="first root ordinal (10000 continues the 10k corpus)")
    parser.add_argument("--roots", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--binstamp", required=True)
    parser.add_argument("--pool-object", default="fl14_v1.jfl1")
    parser.add_argument("--seed", default="3238398113")
    parser.add_argument("--draws", type=int, default=96)
    parser.add_argument("--pilot", type=int, default=32)
    parser.add_argument("--pilot-keep", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=250)
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
             "--filter", f"name~^t2l-{args.run_id}-", "--format", "value(name)"],
            capture_output=True, text=True)
        alive = {l.strip() for l in done.stdout.splitlines() if l.strip()}

    made = 0
    for index in range(args.shards):
        start = args.start + index * per_shard
        count = min(per_shard, args.start + args.roots - start)
        if count <= 0:
            break
        name = f"t2l-{args.run_id}-{start:08d}"
        if name in alive:
            continue
        zone = zones[index]
        metadata = ",".join([
            f"t2l-bucket={BUCKET}", f"t2l-prefix={PREFIX}",
            f"t2l-run-id={args.run_id}", f"t2l-start={start}", f"t2l-count={count}",
            f"t2l-binstamp={args.binstamp}",
            f"t2l-pool-object={args.pool_object}",
            f"t2l-seed={args.seed}", f"t2l-draws={args.draws}",
            f"t2l-pilot={args.pilot}", f"t2l-pilot-keep={args.pilot_keep}",
            f"t2l-chunk={args.chunk}",
            f"t2l-watchdog-seconds={args.watchdog_seconds}",
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
             # Without this the worker authenticates but cannot write, and the
             # only symptom is an empty results prefix hours later.
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
    print(f"{made} instances, {per_shard} roots each, "
          f"draws={args.draws} pilot={args.pilot}/{args.pilot_keep}")


if __name__ == "__main__":
    main()
