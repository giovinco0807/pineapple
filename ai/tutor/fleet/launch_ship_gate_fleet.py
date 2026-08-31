"""Launch the ship-gate fleet: 120 challenger vs shipped 96 on fresh classes.

Thin sibling of `launch_fl_mine_fleet` -- same placement plan, machine
economics and quota behaviour; different startup script and metadata.

Shards are contiguous slices of the fresh-class requests file.  Each class
carries its absolute index in the canonical ordering and the worker derives
its seeds from that, so `--start` selects a window and never re-indexes -- a
shard's numbers are the numbers a local run would produce for those classes.

The binary must carry the 120-dim encoder, added after binstamp 20260830b.
The worker checks that functionally against the challenger image rather than
trusting the stamp, so a stale `--binstamp` fails in seconds instead of
producing a shard of refusals.
"""
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path

from ai.tutor.fleet.launch_hu_street_fleet import (
    GCLOUD, BUCKET, PREFIX, placement_plan, disk_type,
)

STARTUP = Path(__file__).resolve().parent / "startup_ship_gate.sh"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--roots", type=int, required=True,
                        help="fresh classes to cover")
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--requests-object", default="fl14_t0_requests_fresh300.jsonl")
    parser.add_argument("--challenger-object", default="t0_120.bin")
    parser.add_argument("--models-object", required=True)
    parser.add_argument("--models-subdir", default="own_lap4")
    parser.add_argument("--src", required=True)
    parser.add_argument("--binstamp", required=True)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--rollouts", type=int, default=256)
    parser.add_argument("--machine-type", default="c4-standard-8")
    parser.add_argument("--reserve-cores", type=int, default=16)
    parser.add_argument("--placement-offset", type=int, default=0)
    parser.add_argument("--watchdog-seconds", type=int, default=10800)
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
             "--filter", f"name~^hu-{args.run_id}-", "--format", "value(name)"],
            capture_output=True, text=True)
        alive = {l.strip() for l in done.stdout.splitlines() if l.strip()}

    made = 0
    for index in range(args.shards):
        start = args.start + index * per_shard
        count = min(per_shard, args.start + args.roots - start)
        if count <= 0:
            break
        name = f"hu-{args.run_id}-gate-{start:06d}"
        if name in alive:
            continue
        zone = zones[index]
        metadata = ",".join([
            f"hu-bucket={BUCKET}", f"hu-prefix={PREFIX}",
            f"hu-run-id={args.run_id}", f"hu-start={start}", f"hu-count={count}",
            f"hu-src={args.src}", f"hu-binstamp={args.binstamp}",
            f"hu-requests-object={args.requests_object}",
            f"hu-challenger-object={args.challenger_object}",
            f"hu-models-object={args.models_object}",
            f"hu-models-subdir={args.models_subdir}",
            f"hu-batches={args.batches}", f"hu-rollouts={args.rollouts}",
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
            print(f"FAILED {name} {zone}: "
                  f"{done.stderr.strip().splitlines()[-1] if done.stderr else '?'}")
    print(f"{made} instances, {per_shard} classes each, "
          f"{args.batches}x{args.rollouts} rollouts per disagreement")


if __name__ == "__main__":
    main()
