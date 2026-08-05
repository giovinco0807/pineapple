"""Launch a joker-track label-generation fleet on GCP Spot instances.

Shards are contiguous root ranges: root seeds are absolute (seed + offset) and
chunk files are named by seed, so a shard's work is defined entirely by
(--seed-base, --roots-per-shard, shard index) and any shard can be rerun or
resumed without coordinating with the others.

The regional Spot CPU quota is per region, and asia-northeast1 is fully
consumed by the regular track's fleet, so this defaults to us-central1.

Usage:
    python -m ai.tutor.fleet.launch_joker_fleet --street t1 --shards 24 \
        --roots-per-shard 500 --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

BUCKET = "pokerhu-ofc-solver-485418-training"
SERVICE_ACCOUNT = "ofc-labelgen-worker@ofc-solver-485418.iam.gserviceaccount.com"
STARTUP = Path(__file__).resolve().parent / "startup_joker_labelgen.sh"
# On Windows the entry point is gcloud.cmd, which subprocess cannot resolve
# from the bare name without a shell.
GCLOUD = shutil.which("gcloud") or "gcloud"


def run(command: list[str], dry_run: bool) -> tuple[int, str]:
    command = [GCLOUD if part == "gcloud" else part for part in command]
    if dry_run:
        print("DRY " + " ".join(command))
        return 0, ""
    completed = subprocess.run(command, capture_output=True, text=True)
    return completed.returncode, (completed.stderr or completed.stdout)[-400:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--street", choices=["t0", "t1", "t2"], required=True)
    parser.add_argument("--run-id", default=None, help="defaults to <street>-r1")
    parser.add_argument("--shards", type=int, default=24)
    parser.add_argument("--roots-per-shard", type=int, default=500)
    parser.add_argument("--seed-base", type=int, default=101_000_000)
    parser.add_argument("--batch", type=int, default=125)
    parser.add_argument("--machine-type", default="c4-standard-8")
    parser.add_argument("--zones", default="us-central1-b,us-central1-c,us-central1-f,us-west1-b,us-west1-c")
    parser.add_argument("--watchdog-seconds", type=int, default=10_800)
    parser.add_argument(
        "--extra-args", default="",
        help="passed through to the generator, e.g. '--t2-samples 20'",
    )
    parser.add_argument(
        "--skip-done", action="store_true",
        help="Skip shards whose done marker already exists (preemption "
        "recovery relaunches only what is missing).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or f"{args.street}-r1"
    prefix = f"joker-fleet/runs/{run_id}"
    zones = args.zones.split(",")
    plan = {
        "run_id": run_id,
        "street": args.street,
        "shards": args.shards,
        "roots_per_shard": args.roots_per_shard,
        "roots_total": args.shards * args.roots_per_shard,
        "seed_base": args.seed_base,
        "prefix": prefix,
        "machine_type": args.machine_type,
        "zones": zones,
    }
    print(json.dumps(plan, indent=2))

    done_seeds: set[str] = set()
    if args.skip_done:
        listing = subprocess.run(
            [GCLOUD, "storage", "ls", f"gs://{BUCKET}/{prefix}/done/"],
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
        if args.skip_done and str(seed) in done_seeds:
            continue
        name = f"jk-{run_id}-{index:02d}"
        metadata = [
            f"jk-bucket={BUCKET}",
            f"jk-prefix={prefix}",
            f"jk-street={args.street}",
            f"jk-seed={seed}",
            f"jk-roots={args.roots_per_shard}",
            f"jk-batch={args.batch}",
            f"jk-extra-args={args.extra_args}",
            f"jk-watchdog-seconds={args.watchdog_seconds}",
        ]
        placed = False
        # Spot pools are per machine family; when every zone is dry for one
        # family, a sibling family usually still has capacity.
        machine_types = [args.machine_type, "n2-standard-8", "e2-standard-8"]
        for machine_type in dict.fromkeys(machine_types):
          for zone in zones:
            code, message = run(
                [
                    "gcloud", "compute", "instances", "create", name,
                    "--zone", zone,
                    "--machine-type", machine_type,
                    "--provisioning-model", "SPOT",
                    "--instance-termination-action", "DELETE",
                    "--image-family", "debian-12",
                    "--image-project", "debian-cloud",
                    "--boot-disk-size", "50GB",
                    "--scopes", "cloud-platform",
                    "--service-account", SERVICE_ACCOUNT,
                    "--metadata", ",".join(metadata),
                    "--metadata-from-file", f"startup-script={STARTUP}",
                ],
                args.dry_run,
            )
            if code == 0:
                launched.append((name, zone, seed))
                placed = True
                break
            # Zonal stockouts are routine for Spot; the next zone is the fix.
            print(f"  {name}: {zone}/{machine_type} unavailable ({message.strip()[:60]})")
          if placed:
            break
        if not placed:
            failed.append((name, seed))

    print(f"\nlaunched {len(launched)} / {args.shards}")
    for name, zone, seed in launched[:5]:
        print(f"  {name} {zone} seed={seed}")
    if failed:
        print(f"FAILED to place {len(failed)} shards:", [name for name, _ in failed])
        sys.exit(1)


if __name__ == "__main__":
    main()
