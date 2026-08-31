"""Launch the vs-FL T0 mining fleet: own-worth referee over a root range.

Thin sibling of `launch_t0_mine_fleet` -- same placement plan, machine
economics, and quota behaviour; different startup script and metadata.
Shards are contiguous ranges of the FREQUENCY order (multiplicity desc, ties
by id) that `ai/tutor/fl_mine_local.py` rebuilds from the requests and
multiplicity files, so a range means the same roots on every worker and the
seed bands, keyed on the absolute index, cannot collide between shards.

The multiplicity object is passed explicitly rather than assumed present:
the driver falls back to an unordered pool when it cannot find that file, and
an unordered pool is not an error anyone would see -- it is 63 shards quietly
mining the wrong roots.  The startup script re-derives the name and refuses
to start if it disagrees with what was staged.
"""
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path

from ai.tutor.fleet.launch_hu_street_fleet import (
    GCLOUD, BUCKET, PREFIX, placement_plan, disk_type,
)

STARTUP = Path(__file__).resolve().parent / "startup_fl_mine.sh"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--start", type=int, default=0,
                        help="first root index in the frequency order")
    parser.add_argument("--roots", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--requests-object", default="fl14_t0_requests_v1.jsonl")
    parser.add_argument("--multiplicity-object", default=None,
                        help="defaults to the requests name with "
                             ".multiplicity.json, which is what the driver derives")
    parser.add_argument("--models-object", required=True)
    parser.add_argument("--models-subdir", default="own_lap4",
                        help="the chain the referee plays; the HU pair models "
                             "are never loaded on this path")
    parser.add_argument("--src", required=True)
    parser.add_argument("--binstamp", required=True)
    parser.add_argument("--machine-type", default="c4-standard-8")
    parser.add_argument("--reserve-cores", type=int, default=16)
    parser.add_argument("--placement-offset", type=int, default=0)
    parser.add_argument("--watchdog-seconds", type=int, default=18000)
    parser.add_argument("--skip-done", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and create nothing")
    args = parser.parse_args()

    multiplicity = args.multiplicity_object or Path(
        args.requests_object).with_suffix(".multiplicity.json").name

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
             "--filter", f"name~^hu-{args.run_id}-", "--format", "value(name)"],
            capture_output=True, text=True)
        alive = {l.strip() for l in done.stdout.splitlines() if l.strip()}

    made = 0
    for index in range(args.shards):
        start = args.start + index * per_shard
        count = min(per_shard, args.start + args.roots - start)
        if count <= 0:
            break
        name = f"hu-{args.run_id}-mine-{start:06d}"
        if name in alive:
            continue
        zone = zones[index]
        metadata = ",".join([
            f"hu-bucket={BUCKET}", f"hu-prefix={PREFIX}",
            f"hu-run-id={args.run_id}", f"hu-start={start}", f"hu-count={count}",
            f"hu-src={args.src}", f"hu-binstamp={args.binstamp}",
            f"hu-requests-object={args.requests_object}",
            f"hu-multiplicity-object={multiplicity}",
            f"hu-models-object={args.models_object}",
            f"hu-models-subdir={args.models_subdir}",
            f"hu-watchdog-seconds={args.watchdog_seconds}",
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
            print(f"FAILED {name} {zone}: {done.stderr.strip().splitlines()[-1] if done.stderr else '?'}")
    print(f"{made} instances, {per_shard} roots each, multiplicity={multiplicity}")


if __name__ == "__main__":
    main()
