"""Launch the T1-vs-FL label fleet on GCP Spot instances.

A shard is a contiguous range of absolute root seeds.  Roots are dealt from
their seed alone, labels are published one object per root named by that seed,
and every upload is create-only -- so a shard's work is defined entirely by
`(--seed-base, --roots-per-shard, index)`, any shard can be re-run without
coordinating with the others, and `--skip-done` relaunches only what is
missing after a wave of preemptions.

# Placement

Spot capacity is per (zone, machine family) and Spot CPU quota is per region,
so shards are spread across zones inside a region and regions are filled
cheapest-first.  Measured Spot price for c4-standard-8: europe-west4 $0.165/h,
us-west1 $0.237/h, Tokyo $0.293/h.  Measured Spot CPU quota, in cores:
us-central1 258, us-west1 252, europe-west4 100 -- which at 8 vCPU a worker is
32 / 31 / 12 instances, and the cap this respects.

When one family is dry in every zone of a region, a sibling family usually
still has capacity, so each shard falls back c4 -> n2 -> e2.  C4 boots only on
Hyperdisk, N2/E2 only on persistent disks, so the disk type follows the
family rather than being a constant.

# Deletion

`--instance-termination-action DELETE` covers preemption, `--max-run-duration`
covers a worker that hangs, and the worker deletes itself when its shard is
done.  All three exist because a Spot VM that merely *stops* keeps billing for
its disk: 84 stopped-but-undeleted instances cost this project ~$10 once.

Usage:
    python -m ai.tutor.fleet.launch_t1_vs_fl_fleet --run-id t1-r1 \
        --shards 30 --roots-per-shard 100 --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

BUCKET = "pokerhu-ofc-solver-485418-training"
PREFIX = "fl14-t1"
SERVICE_ACCOUNT = "ofc-labelgen-worker@ofc-solver-485418.iam.gserviceaccount.com"
STARTUP = Path(__file__).resolve().parent / "startup_t1_vs_fl.sh"
# On Windows the entry point is gcloud.cmd, which subprocess cannot resolve
# from the bare name without a shell.
GCLOUD = shutil.which("gcloud") or "gcloud"

# Cheapest first; the cores are the measured regional Spot CPU quota.
REGIONS: list[tuple[str, list[str], int]] = [
    ("europe-west4", ["europe-west4-a", "europe-west4-b", "europe-west4-c"], 100),
    ("us-west1", ["us-west1-a", "us-west1-b", "us-west1-c"], 252),
    ("us-central1",
     ["us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f"], 258),
]
FAMILY_DISK = {"c4": "hyperdisk-balanced"}
DEFAULT_DISK = "pd-balanced"


def disk_type(machine_type: str) -> str:
    return FAMILY_DISK.get(machine_type.split("-", 1)[0], DEFAULT_DISK)


def vcpus(machine_type: str) -> int:
    return int(machine_type.rsplit("-", 1)[-1])


def placement_plan(shards: int, machine_type: str, reserve: int) -> list[str]:
    """One zone per shard: round-robin inside a region, cheapest region first."""
    cores = vcpus(machine_type)
    plan: list[str] = []
    for _, zones, quota in REGIONS:
        capacity = max((quota - reserve) // cores, 0)
        for index in range(capacity):
            if len(plan) >= shards:
                return plan
            plan.append(zones[index % len(zones)])
    return plan


def run(command: list[str], dry_run: bool) -> tuple[int, str]:
    if dry_run:
        print("DRY " + " ".join(command))
        return 0, ""
    done = subprocess.run(command, capture_output=True, text=True)
    return done.returncode, (done.stderr or done.stdout)[-400:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--roots-per-shard", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=201_000_000)
    parser.add_argument("--machine-type", default="c4-standard-8")
    parser.add_argument(
        "--reserve-cores", type=int, default=16,
        help="regional Spot cores to leave for other fleets",
    )
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument("--src", default="t1_src_20260812c.tar.gz")
    parser.add_argument("--binstamp", default="20260812a")
    parser.add_argument("--pool-object", default="fl14_v1.jfl1")
    parser.add_argument("--t2-object", default="t2_evaluator.bin")
    parser.add_argument("--t3-object", default="t3_evaluator.bin")
    parser.add_argument("--publish-seconds", type=int, default=30)
    parser.add_argument("--watchdog-seconds", type=int, default=21_600)
    parser.add_argument(
        "--extra-args", default="",
        help="passed to run_t1_shard, e.g. '--t2-samples 32 --pool-opponents 1500'",
    )
    parser.add_argument(
        "--skip-done", action="store_true",
        help="skip shards whose done marker exists (preemption recovery "
             "relaunches only what is missing)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    zones = placement_plan(args.shards, args.machine_type, args.reserve_cores)
    if len(zones) < args.shards:
        raise SystemExit(
            f"quota fits {len(zones)} worker(s) of {args.machine_type}, "
            f"{args.shards} asked for"
        )
    print(json.dumps(
        {
            "run_id": args.run_id,
            "shards": args.shards,
            "roots_per_shard": args.roots_per_shard,
            "roots_total": args.shards * args.roots_per_shard,
            "seed_base": args.seed_base,
            "labels": f"gs://{args.bucket}/{args.prefix}/runs/{args.run_id}/labels/",
            "machine_type": args.machine_type,
            "zones": sorted(set(zones)),
            "extra_args": args.extra_args,
        },
        indent=2,
    ))

    done_seeds: set[str] = set()
    if args.skip_done:
        listing = subprocess.run(
            [GCLOUD, "storage", "ls",
             f"gs://{args.bucket}/{args.prefix}/runs/{args.run_id}/done/"],
            capture_output=True, text=True,
        )
        for line in listing.stdout.splitlines():
            marker = line.rsplit("/", 1)[-1].removesuffix(".txt").strip()
            if marker:
                done_seeds.add(marker)
        print(f"skip-done: {len(done_seeds)} shard(s) already complete")

    launched, failed = [], []
    for index in range(args.shards):
        seed = args.seed_base + index * args.roots_per_shard
        if str(seed) in done_seeds:
            continue
        # The run id carries the street, so the instance name is just
        # `<run-id>-<shard>`; it must be a legal RFC1035 name.
        name = f"{args.run_id}-{index:02d}"
        metadata = [
            f"t1-bucket={args.bucket}",
            f"t1-prefix={args.prefix}",
            f"t1-run-id={args.run_id}",
            f"t1-seed={seed}",
            f"t1-roots={args.roots_per_shard}",
            f"t1-src={args.src}",
            f"t1-binstamp={args.binstamp}",
            f"t1-pool-object={args.pool_object}",
            f"t1-t2-object={args.t2_object}",
            f"t1-t3-object={args.t3_object}",
            f"t1-publish-seconds={args.publish_seconds}",
            f"t1-watchdog-seconds={args.watchdog_seconds}",
            f"t1-extra-args={args.extra_args}",
        ]
        home = zones[index]
        region = home.rsplit("-", 1)[0]
        placed = False
        for machine_type in ("c4-standard-8", "n2-standard-8", "e2-standard-8") \
                if args.machine_type == "c4-standard-8" else (args.machine_type,):
            # Stay inside the shard's own region so the plan's quota
            # arithmetic keeps holding, but try its sibling zones.
            zone_options = next(z for name_, z, _ in REGIONS if name_ == region)
            ordered = [home] + [z for z in zone_options if z != home]
            for zone in ordered:
                code, message = run(
                    [
                        GCLOUD, "compute", "instances", "create", name,
                        "--zone", zone,
                        "--machine-type", machine_type,
                        "--provisioning-model", "SPOT",
                        "--instance-termination-action", "DELETE",
                        # The platform's own backstop: a worker that wedges
                        # before its watchdog can fire is still deleted.
                        "--max-run-duration", f"{args.watchdog_seconds + 1800}s",
                        "--image-family", "debian-12",
                        "--image-project", "debian-cloud",
                        "--boot-disk-size", "50GB",
                        "--boot-disk-type", disk_type(machine_type),
                        "--scopes", "cloud-platform",
                        "--service-account", SERVICE_ACCOUNT,
                        "--metadata", ",".join(metadata),
                        "--metadata-from-file", f"startup-script={STARTUP}",
                    ],
                    args.dry_run,
                )
                if code == 0:
                    launched.append((name, zone, machine_type, seed))
                    placed = True
                    break
                print(f"  {name}: {zone}/{machine_type} "
                      f"unavailable ({message.strip()[:80]})")
            if placed:
                break
        if not placed:
            failed.append((name, seed))

    print(f"\nlaunched {len(launched)} / {args.shards}")
    for name, zone, machine_type, seed in launched[:8]:
        print(f"  {name} {zone} {machine_type} seed={seed}")
    if failed:
        print(f"FAILED to place {len(failed)}:", [name for name, _ in failed])
        sys.exit(1)


if __name__ == "__main__":
    main()
