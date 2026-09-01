"""Launch the HU trace fleet: champion-vs-champion hands as teaching material.

Each shard plays `--hands` hands under its own `--self-play-seed`, so shards
never overlap and a lost shard is replaced by relaunching that index alone.
Seeds are `seed_base + shard * stride`; the stride is large enough that two
shards cannot walk into each other's hand numbering.
"""
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path

from ai.tutor.fleet.launch_hu_street_fleet import (
    GCLOUD, BUCKET, PREFIX, placement_plan, disk_type,
)

STARTUP = Path(__file__).resolve().parent / "startup_hu_trace.sh"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--hands", type=int, required=True, help="per shard")
    parser.add_argument("--seed-base", type=int, default=8_888_000_001)
    parser.add_argument("--seed-stride", type=int, default=1_000_003)
    parser.add_argument("--binstamp", required=True)
    parser.add_argument("--models-object", default="models_ship_20260903.tar.gz")
    parser.add_argument("--chunk", type=int, default=200)
    parser.add_argument("--machine-type", default="n2-highcpu-32")
    parser.add_argument("--reserve-cores", type=int, default=16)
    parser.add_argument("--placement-offset", type=int, default=0)
    parser.add_argument("--watchdog-seconds", type=int, default=14400)
    parser.add_argument("--skip-done", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    zones = placement_plan(
        args.shards + args.placement_offset, args.machine_type, args.reserve_cores
    )[args.placement_offset:]
    if len(zones) < args.shards:
        raise SystemExit(f"quota fits {len(zones)} instances, {args.shards} asked for")

    alive: set[str] = set()
    if args.skip_done:
        done = subprocess.run(
            [GCLOUD, "compute", "instances", "list",
             "--filter", f"name~^ht-{args.run_id}-", "--format", "value(name)"],
            capture_output=True, text=True)
        alive = {l.strip() for l in done.stdout.splitlines() if l.strip()}

    made = 0
    for shard in range(args.shards):
        name = f"ht-{args.run_id}-{shard:04d}"
        if name in alive:
            continue
        zone = zones[shard]
        metadata = ",".join([
            f"ht-bucket={BUCKET}", f"ht-prefix={PREFIX}",
            f"ht-run-id={args.run_id}", f"ht-shard={shard}",
            f"ht-hands={args.hands}",
            f"ht-seed={args.seed_base + shard * args.seed_stride}",
            f"ht-binstamp={args.binstamp}",
            f"ht-models-object={args.models_object}",
            f"ht-chunk={args.chunk}",
            f"ht-watchdog-seconds={args.watchdog_seconds}",
        ])
        if args.dry_run:
            print(f"would launch {name} {zone}")
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
                print(f"launched {name} {zone} seed "
                      f"{args.seed_base + shard * args.seed_stride}")
        else:
            print(f"FAILED {name} {zone}: "
                  f"{done.stderr.strip().splitlines()[-1] if done.stderr else '?'}")
    print(f"{made} instances x {args.hands} hands = {made * args.hands} traced")


if __name__ == "__main__":
    main()
