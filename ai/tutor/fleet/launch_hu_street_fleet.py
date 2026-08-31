"""Launch the HU street-teacher fleet on GCP Spot instances.

A shard is a contiguous slice of one street's requests file, and the file
ships to every worker, so a shard's work is defined entirely by
`(job, start, count)`.  Labels are published one object per root, named by an
id unique across the job, create-only -- so any shard can be re-run without
coordinating with the others and `--skip-done` relaunches only what a wave of
preemptions left missing.

# Placement

Spot capacity is per (zone, machine family) and Spot CPU quota is per region,
so shards spread across zones inside a region and regions fill cheapest-first.
Measured Spot price for c4-standard-8: europe-west4 $0.165/h, us-west1
$0.237/h, Tokyo $0.293/h.  Measured Spot CPU quota, in cores: us-central1 258,
us-west1 252, europe-west4 100 -- at 8 vCPU a worker, 32 / 31 / 12 instances.

# Deletion

`--instance-termination-action DELETE` covers preemption, `--max-run-duration`
covers a worker that hangs, and the worker deletes itself when its shard is
done.  All three exist because a Spot VM that merely *stops* keeps billing for
its disk: 84 stopped-but-undeleted instances cost this project ~$10 once.

Usage:
    python -m ai.tutor.fleet.launch_hu_street_fleet --run-id t0-r1 \\
        --job t0_btn --seat btn --roots 10000 --shards 40 \\
        --requests-object t0_btn_requests.jsonl --value-object v1s.npz --dry-run
"""
from __future__ import annotations

import argparse
import concurrent.futures
import shutil
import subprocess
import tempfile
from pathlib import Path

BUCKET = "pokerhu-ofc-solver-485418-training"
PREFIX = "hu-street"
SERVICE_ACCOUNT = "ofc-labelgen-worker@ofc-solver-485418.iam.gserviceaccount.com"
STARTUP = Path(__file__).resolve().parent / "startup_hu_street.sh"
# On Windows the entry point is gcloud.cmd, which subprocess cannot resolve
# from the bare name without a shell.
GCLOUD = shutil.which("gcloud") or "gcloud"

# Cheapest first; the cores are the measured regional Spot CPU quota.
REGIONS: list[tuple[str, list[str], int]] = [
    ("europe-west4", ["europe-west4-b", "europe-west4-c"], 100),
    ("us-west1", ["us-west1-a", "us-west1-b", "us-west1-c"], 252),
    ("us-central1",
     ["us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f"], 258),
]
FAMILY_DISK = {"c4": "hyperdisk-balanced"}
DEFAULT_DISK = "pd-balanced"
FALLBACK = ["c4-standard-8", "n2-standard-8", "e2-standard-8"]


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


def existing(prefix: str) -> set[str]:
    done = subprocess.run(
        [GCLOUD, "compute", "instances", "list",
         "--filter", f"name~^{prefix}", "--format", "value(name)"],
        capture_output=True, text=True,
    )
    return {line.strip() for line in done.stdout.splitlines() if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--job", required=True, help="e.g. t0_btn")
    parser.add_argument("--seat", choices=["btn", "bb"], required=True)
    parser.add_argument("--roots", type=int, required=True,
                        help="requests to label, from line 0")
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--requests-object", required=True)
    parser.add_argument("--value-object", required=True)
    parser.add_argument("--machine-type", default="c4-standard-8")
    parser.add_argument("--reserve-cores", type=int, default=16,
                        help="regional Spot cores to leave for other fleets")
    parser.add_argument("--placement-offset", type=int, default=0,
                        help="skip this many slots of the placement plan.  Two "
                             "jobs launched together must not both start at "
                             "the cheapest region: Spot CPU quota is regional "
                             "and shared across families, so the second would "
                             "be refused where the first already sits.")
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument("--src", default="hu_src_20260817b.tar.gz")
    parser.add_argument("--binstamp", default="20260817a")
    parser.add_argument("--joint-samples", type=int, default=400)
    parser.add_argument("--batch-roots", type=int, default=8)
    parser.add_argument("--watchdog-seconds", type=int, default=21_600)
    parser.add_argument("--extra-args", default="",
                        help="passed to the teacher, e.g. "
                             "'--no-traced-opponent --opp-draws 4'")
    parser.add_argument("--labels-object", default="",
                        help="a finished labels file in artifacts/; workers "
                             "then only encode")
    parser.add_argument("--encode", action="store_true",
                        help="workers also publish their feature matrices; at "
                             "T0 encoding costs as much as labelling, so "
                             "leaving it at home undoes the point of the fleet")
    parser.add_argument("--skip-done", action="store_true",
                        help="skip shards whose instance is still alive "
                             "(preemption recovery relaunches the rest)")
    parser.add_argument("--parallel", type=int, default=12,
                        help="instance creations in flight.  Serially, each "
                             "create is about eight seconds, so a sixty-eight "
                             "worker fleet spends twenty minutes being born "
                             "-- half as long as it then takes to run.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    per_shard = -(-args.roots // args.shards)
    zones = placement_plan(
        args.shards + args.placement_offset, args.machine_type, args.reserve_cores
    )[args.placement_offset:]
    if len(zones) < args.shards:
        raise SystemExit(
            f"quota fits {len(zones)} instances of {args.machine_type}, "
            f"{args.shards} asked for"
        )
    alive = existing(f"hu-{args.run_id}") if args.skip_done else set()
    holder = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                         encoding="utf-8")
    holder.write(args.extra_args)
    holder.close()
    extra_path = holder.name

    def create(index: int, zone: str) -> tuple[str, str | None]:
        """Make one shard's instance, falling back across families."""
        start = index * per_shard
        count = min(per_shard, args.roots - start)
        name = f"hu-{args.run_id}-{args.job.replace('_', '-')}-{start:07d}"
        if name in alive:
            return name, "still running"
        metadata = ",".join([
            f"hu-bucket={args.bucket}",
            f"hu-prefix={args.prefix}",
            f"hu-run-id={args.run_id}",
            f"hu-job={args.job}",
            f"hu-seat={args.seat}",
            f"hu-start={start}",
            f"hu-count={count}",
            f"hu-src={args.src}",
            f"hu-binstamp={args.binstamp}",
            f"hu-requests-object={args.requests_object}",
            f"hu-value-object={args.value_object}",
            f"hu-joint-samples={args.joint_samples}",
            f"hu-batch-roots={args.batch_roots}",
            f"hu-watchdog-seconds={args.watchdog_seconds}",
            f"hu-encode={'--encode' if args.encode else ''}",
            f"hu-labels-object={args.labels_object}",
        ])
        for machine_type in [args.machine_type] + [
            m for m in FALLBACK if m != args.machine_type
        ]:
            command = [
                GCLOUD, "compute", "instances", "create", name,
                "--zone", zone,
                "--machine-type", machine_type,
                "--provisioning-model", "SPOT",
                "--instance-termination-action", "DELETE",
                "--max-run-duration", f"{args.watchdog_seconds + 1800}s",
                "--boot-disk-type", disk_type(machine_type),
                "--boot-disk-size", "50GB",
                "--image-family", "debian-12",
                "--image-project", "debian-cloud",
                "--scopes", "cloud-platform",
                "--service-account", SERVICE_ACCOUNT,
                # extra-args travels as a file: it contains spaces, and the
                # Windows gcloud entry point is a .CMD that re-splits argv on
                # them -- the documented cause of this fleet's silent
                # empty-stdout failures once already.
                "--metadata-from-file",
                f"startup-script={STARTUP},hu-extra-args={extra_path}",
                "--metadata", metadata,
            ]
            code, blurb = run(command, args.dry_run)
            if code == 0:
                return name, f"{zone} {machine_type} [{start}, +{count})"
            print(f"  {name}: {machine_type} in {zone} refused: "
                  f"{blurb.strip()[:140]}", flush=True)
        return name, None

    plan = [(index, zone) for index, zone in enumerate(zones)
            if index * per_shard < args.roots]
    launched = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for name, verdict in pool.map(lambda pair: create(*pair), plan):
            if verdict is None:
                print(f"FAILED {name}: no family had capacity")
            elif verdict == "still running":
                print(f"skip {name}: still running")
            else:
                print(f"launched {name} {verdict}")
                launched += 1
    print(f"{launched} instances, {per_shard} roots each")


if __name__ == "__main__":
    main()
