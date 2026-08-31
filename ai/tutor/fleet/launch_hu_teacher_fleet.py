"""Launch a Rust-teacher labelling job on GCP Spot instances.

A shard is a line range of one requests file, like the street fleet -- but
the worker runs a `t4_first_exact` labelling mode directly and publishes one
object per shard, like the match fleet, because Rust teachers write whole
files and finish shards in minutes.  Naming follows the street fleet's
`(job, start)` convention exactly so `reap_hu_fleet` monitors this fleet
unchanged: instances are `hu-{run}-{job}-{start:07d}` and clean exits mark
`{job}_{start}_generate_ok.txt`.

Usage:
    python -m ai.tutor.fleet.launch_hu_teacher_fleet --run-id relabel1 \\
        --job t3bb16 --mode t3-first-hu --roots 100000 --shards 64 \\
        --requests-object t3_bb_req_d16.jsonl --chooser-object ranker96.bin
"""
from __future__ import annotations

import argparse
import concurrent.futures

from ai.tutor.fleet.launch_hu_street_fleet import (
    BUCKET, FALLBACK, GCLOUD, PREFIX, SERVICE_ACCOUNT, disk_type,
    existing, placement_plan, run,
)
from pathlib import Path

STARTUP = Path(__file__).resolve().parent / "startup_hu_rust_teacher.sh"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--job", required=True,
                        help="label namespace; no underscores beyond the "
                             "job/start split the reaper parses, e.g. t3bb16")
    parser.add_argument("--mode", required=True,
                        help="binary mode without leading dashes, "
                             "e.g. t3-first-hu")
    parser.add_argument("--roots", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--requests-object", required=True)
    parser.add_argument("--chooser-object", required=True)
    parser.add_argument("--machine-type", default="c4-standard-8")
    parser.add_argument("--reserve-cores", type=int, default=16)
    parser.add_argument("--placement-offset", type=int, default=0)
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument("--src", default="hu_src_20260817h.tar.gz")
    parser.add_argument("--binstamp", default="20260817b")
    parser.add_argument("--watchdog-seconds", type=int, default=5400)
    parser.add_argument("--parallel", type=int, default=12)
    parser.add_argument("--skip-done", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if "_" in args.job:
        raise SystemExit("job names with underscores break the reaper's "
                         "marker parsing; use e.g. t3bb16")
    per_shard = -(-args.roots // args.shards)
    zones = placement_plan(
        args.shards + args.placement_offset, args.machine_type,
        args.reserve_cores,
    )[args.placement_offset:]
    if len(zones) < args.shards:
        raise SystemExit(f"quota fits {len(zones)}, asked {args.shards}")
    alive = existing(f"hu-{args.run_id}") if args.skip_done else set()

    def create(index: int, zone: str) -> tuple[str, str | None]:
        start = index * per_shard
        count = min(per_shard, args.roots - start)
        name = f"hu-{args.run_id}-{args.job}-{start:07d}"
        if name in alive:
            return name, "still running"
        metadata = ",".join([
            f"hu-bucket={args.bucket}",
            f"hu-prefix={args.prefix}",
            f"hu-run-id={args.run_id}",
            f"hu-job={args.job}",
            f"hu-mode={args.mode}",
            f"hu-start={start}",
            f"hu-count={count}",
            f"hu-src={args.src}",
            f"hu-binstamp={args.binstamp}",
            f"hu-requests-object={args.requests_object}",
            f"hu-chooser-object={args.chooser_object}",
            f"hu-watchdog-seconds={args.watchdog_seconds}",
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
                "--metadata-from-file", f"startup-script={STARTUP}",
                "--metadata", metadata,
            ]
            code, blurb = run(command, args.dry_run)
            if code == 0:
                return name, f"{zone} {machine_type} [{start}, +{count})"
            print(f"  {name}: {machine_type} in {zone} refused: "
                  f"{blurb.strip()[:140]}", flush=True)
        return name, None

    plan = [(i, z) for i, z in enumerate(zones) if i * per_shard < args.roots]
    launched = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for name, verdict in pool.map(lambda pair: create(*pair), plan):
            if verdict is None:
                print(f"FAILED {name}: no family had capacity")
            elif verdict == "still running":
                print(f"skip {name}")
            else:
                print(f"launched {name} {verdict}")
                launched += 1
    print(f"{launched} instances, {per_shard} roots each")


if __name__ == "__main__":
    main()
