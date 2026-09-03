"""Launch the T0-BB mining fleet: shape-aware referee over a root range.

Thin sibling of `launch_hu_street_fleet` -- same placement plan, machine
economics, and quota behaviour; different startup script and metadata.
Shards are root ranges of the SHUFFLED requests order (shuffle seed fixed
inside ai/tutor/t0_mine.py), so ranges never overlap across shards or with
the local Phase-0 runs, which used indices 0..71.
"""
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path

from ai.tutor.fleet.launch_hu_street_fleet import (
    GCLOUD, BUCKET, PREFIX, placement_plan, disk_type,
)

STARTUP = Path(__file__).resolve().parent / "startup_t0_label.sh"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--start", type=int, required=True,
                        help="first root index in the shuffled order")
    parser.add_argument("--roots", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--requests-object", default="t0_bb_onpol.jsonl")
    parser.add_argument("--rollouts", type=int, default=128)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--seat", choices=("bb", "btn"), default="btn",
                        help="btn: requests object is roots_all.jsonl format "
                             "(btn_cards/bb_board/served); passed to t0_mine.py --seat")
    parser.add_argument("--models-object", required=True)
    parser.add_argument("--src", required=True)
    parser.add_argument("--binstamp", required=True)
    parser.add_argument("--machine-type", default="c4-standard-8")
    parser.add_argument("--reserve-cores", type=int, default=16)
    parser.add_argument("--placement-offset", type=int, default=0)
    parser.add_argument("--watchdog-seconds", type=int, default=18000)
    parser.add_argument("--skip-done", action="store_true")
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
        name = f"hu-{args.run_id}-label-{start:06d}"
        if name in alive:
            continue
        zone = zones[index]
        metadata = ",".join([
            f"hu-bucket={BUCKET}", f"hu-prefix={PREFIX}",
            f"hu-run-id={args.run_id}", f"hu-start={start}", f"hu-count={count}",
            f"hu-src={args.src}", f"hu-binstamp={args.binstamp}",
            f"hu-requests-object={args.requests_object}",
            f"hu-models-object={args.models_object}",
            f"hu-watchdog-seconds={args.watchdog_seconds}",
            f"hu-seat={args.seat}", f"hu-rollouts={args.rollouts}", f"hu-passes={args.passes}",
        ])
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
            print(f"FAILED {name} {zone}: {done.stderr.strip().splitlines()[-1] if done.stderr else '?'}")
    print(f"{made} instances, {per_shard} roots each")


if __name__ == "__main__":
    main()
