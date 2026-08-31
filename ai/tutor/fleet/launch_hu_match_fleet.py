"""Launch a heads-up match on GCP Spot instances.

A shard is `(job, index)`: its own seed, its own stream of hands, its own
output file.  Sessions carry stacks and Fantasyland state forward inside a
shard, so a shard cannot be resumed halfway -- but it can be replayed from
its seed, which is the recovery procedure for a preemption and the reason
nothing is shared between shards.

Splitting a match across machines does not bias it.  Each shard starts a
fresh session with full stacks, so the only difference from one long run is
that sessions cannot span shard boundaries; per-hand settlement, which is
the measurement, is unaffected.

Placement, deletion and pricing follow `launch_hu_street_fleet`; see its
docstring for why each of those is shaped the way it is.

Usage:
    python -m ai.tutor.fleet.launch_hu_match_fleet --run-id m1 --job vs_lap4 \\
        --hands 5000 --shards 34 --models-object models_20260817.tar.gz \\
        --match-args "--hu-a-models ... --arm-b-own ..." --dry-run
"""
from __future__ import annotations

import argparse
import concurrent.futures
import shutil
import subprocess
import tempfile
from pathlib import Path

from ai.tutor.fleet.launch_hu_street_fleet import (
    BUCKET, FALLBACK, PREFIX, SERVICE_ACCOUNT, disk_type, placement_plan, run,
)

GCLOUD = shutil.which("gcloud") or "gcloud"
STARTUP = Path(__file__).resolve().parent / "startup_hu_match.sh"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--job", required=True, help="e.g. vs_lap4")
    parser.add_argument("--hands", type=int, required=True,
                        help="hands in the whole job, split across shards")
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--models-object", required=True)
    parser.add_argument("--match-args", required=True,
                        help="the arm flags, with paths relative to the "
                             "worker's root (models/...)")
    parser.add_argument("--seed-base", type=int, default=20_260_817)
    parser.add_argument("--machine-type", default="c4-standard-8")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reserve-cores", type=int, default=16)
    parser.add_argument("--placement-offset", type=int, default=0)
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument("--src", default="hu_src_20260817g.tar.gz")
    parser.add_argument("--binstamp", default="20260817b")
    parser.add_argument("--watchdog-seconds", type=int, default=14_400)
    parser.add_argument("--parallel", type=int, default=12)
    parser.add_argument("--skip-done", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    per_shard = -(-args.hands // args.shards)
    zones = placement_plan(
        args.shards + args.placement_offset, args.machine_type, args.reserve_cores
    )[args.placement_offset:]
    if len(zones) < args.shards:
        raise SystemExit(
            f"quota fits {len(zones)} instances, {args.shards} asked for"
        )
    alive: set[str] = set()
    if args.skip_done:
        done = subprocess.run(
            [GCLOUD, "compute", "instances", "list",
             "--filter", f"name~^hm-{args.run_id}-", "--format", "value(name)"],
            capture_output=True, text=True,
        )
        alive = {l.strip() for l in done.stdout.splitlines() if l.strip()}

    # The arm flags are a comma-separated list of model paths, and gcloud's
    # `--metadata` splits values on commas, so they travel as a file.
    holder = tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    )
    holder.write(args.match_args)
    holder.close()
    args_path = holder.name

    def create(index: int, zone: str) -> tuple[str, str | None]:
        hands = min(per_shard, args.hands - index * per_shard)
        name = f"hm-{args.run_id}-{args.job.replace('_', '-')}-{index:03d}"
        if name in alive:
            return name, "still running"
        metadata = ",".join([
            f"hu-bucket={args.bucket}",
            f"hu-prefix={args.prefix}",
            f"hu-run-id={args.run_id}",
            f"hu-job={args.job}",
            f"hu-shard={index}",
            f"hu-hands={hands}",
            # Distinct streams, so shards never replay each other's deals.
            f"hu-seed={args.seed_base + index * 1_000_003}",
            f"hu-src={args.src}",
            f"hu-binstamp={args.binstamp}",
            f"hu-models-object={args.models_object}",
            f"hu-workers={args.workers}",
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
                # The arm flags are a comma-separated list of model paths, and
                # `--metadata` splits on commas -- passed inline, every model
                # after the first becomes a bogus metadata key.
                "--metadata-from-file",
                f"startup-script={STARTUP},hu-match-args={args_path}",
                "--metadata", metadata,
            ]
            code, blurb = run(command, args.dry_run)
            if code == 0:
                return name, f"{zone} {machine_type} {hands} hands"
            print(f"  {name}: {machine_type} in {zone} refused: "
                  f"{blurb.strip()[:140]}", flush=True)
        return name, None

    plan = [(i, z) for i, z in enumerate(zones) if i * per_shard < args.hands]
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
    print(f"{launched} instances, {per_shard} hands each")


if __name__ == "__main__":
    main()
