"""Offline plan and read-only dry run for label-generation cloud runs.

Everything here is either offline arithmetic or a read-only GET through the
existing production adapters. Planning derives, for one run name, the exact
staged object names, their digests, the per-shard instance metadata values
(including the canonical content-bindings the startup script verifies), and
the startup script's own digest -- so what a later execute phase would submit
is fixed and reviewable before any credential exists. The dry run then walks
the same read-only preflight the dataset transport uses: bucket object
metadata for every staged name and the machine-type catalog entry. With no
token in the environment it refuses; with an unusable one it surfaces the
HTTP 401 -- which is the proof that the request construction, not the
authentication, is what this harness exercises.

Instance creation and IAM are deliberately absent. Those go through the
existing lifecycle modules at execute time, under an explicit spend approval,
and nothing in this module can be coaxed into a write: the only adapter
methods it calls are metadata GETs.

Credentials: GOOGLE_OAUTH_ACCESS_TOKEN, read from the environment into the
adapter's memory, never persisted, mirroring the established contract.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import pathlib
import sys
from typing import Any, Mapping

from . import hu_rl_c4_gcp_lifecycle as c4
from .hu_m31_label_gen_worker_v1 import canonical_bytes, load_plan, sha256_of

PLAN_RECEIPT_SCHEMA = "hu_m31_label_gen_run_plan_v1"
# Where a run's instances land. Absent an explicit choice this is the single
# region every earlier run used, so an unflagged invocation plans exactly what
# it always planned.
DEFAULT_ZONE = c4.ZONE
CARRY_MANIFEST_RELATIVE = "static/manifest/carry_manifest.txt"
DEFAULT_BUCKET = "pokerhu-ofc-solver-485418-training"
DEFAULT_MACHINE_TYPE = "c4-standard-4"
DEFAULT_WATCHDOG_SECONDS = 6 * 3600
# One worker process per vCPU: the evaluator is single-threaded, so anything
# less leaves cores idle at the same hourly price.
DEFAULT_WORKER_COUNT = 4
STARTUP_RELATIVE = "scripts/startup_hu_m31_label_gen_v1.sh"


def repository_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def startup_script_sha256() -> str:
    return sha256_of(repository_root() / STARTUP_RELATIVE)


def _binding(relative: str, object_name: str, path: pathlib.Path) -> dict[str, Any]:
    return {
        "relative": relative,
        "object": object_name,
        # Generations exist only after staging; the plan records a placeholder
        # that the stage phase must replace with the observed generation before
        # any instance metadata is rendered.
        "generation": "TO_BE_BOUND_AT_STAGE",
        "sha256": sha256_of(path),
        "bytes": path.stat().st_size,
    }


def build_run_plan(
    *,
    run_name: str,
    package_dir: pathlib.Path,
    plan_file: pathlib.Path,
    bucket: str = DEFAULT_BUCKET,
    machine_type: str = DEFAULT_MACHINE_TYPE,
    watchdog_seconds: int = DEFAULT_WATCHDOG_SECONDS,
    worker_count: int = DEFAULT_WORKER_COUNT,
    zone: str = DEFAULT_ZONE,
    carry_manifest: pathlib.Path | None = None,
) -> dict[str, Any]:
    if not run_name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in run_name):
        raise SystemExit("run name must be lowercase alphanumerics and dashes")
    if zone not in c4.APPROVED_ZONES:
        raise SystemExit(
            f"zone {zone!r} is not one of the approved targets "
            f"{list(c4.APPROVED_ZONES)}; a zone is approved once it is known to "
            "offer the c4 family, hyperdisk-balanced, and the default subnet"
        )
    region = zone.rsplit("-", 1)[0]
    worker_plan = load_plan(plan_file)
    plan_sha = hashlib.sha256(plan_file.read_bytes()).hexdigest()
    prefix = f"labelgen/{run_name}"
    staging = f"{prefix}/staging"

    runtime = package_dir / "runtime.tar.gz"
    wheelhouse = package_dir / "wheelhouse.zip"
    for path in (runtime, wheelhouse):
        if not path.is_file():
            raise SystemExit(f"package artifact missing: {path}")

    bindings = [
        _binding("static/runtime_archive/runtime.tar.gz",
                 f"{staging}/runtime.tar.gz", runtime),
        _binding("static/wheelhouse_archive/wheelhouse.zip",
                 f"{staging}/wheelhouse.zip", wheelhouse),
        _binding("static/plan/plan.json", f"{staging}/plan.json", plan_file),
    ]
    if carry_manifest is not None:
        # Positions an earlier run already published. They travel as an ordinary
        # content binding, so the startup script's existing verify-by-digest
        # fetch loop carries them with no special case, and the workers skip
        # them through the same --existing-manifest path a resumed shard uses.
        if not carry_manifest.is_file():
            raise SystemExit(f"carry manifest missing: {carry_manifest}")
        bindings.append(_binding(
            CARRY_MANIFEST_RELATIVE,
            f"{staging}/carry_manifest.txt",
            carry_manifest,
        ))

    shards = []
    for entry in worker_plan["shards"]:
        shard_id = entry["shard_id"]
        shards.append({
            "shard_id": shard_id,
            "start": entry["start"],
            "count": entry["count"],
            "object_prefix": f"{prefix}/shards/{shard_id}",
            "metadata_values": {
                "lg-bucket": bucket,
                "lg-shard-id": shard_id,
                "lg-object-prefix": f"{prefix}/shards/{shard_id}",
                "lg-plan-sha256": plan_sha,
                "lg-watchdog-seconds": str(watchdog_seconds),
                "lg-worker-count": str(worker_count),
                # lg-attempt-id and lg-content-bindings-b64 (with observed
                # generations) are rendered by the execute phase.
            },
        })

    receipt = {
        "schema": PLAN_RECEIPT_SCHEMA,
        "run_name": run_name,
        "bucket": bucket,
        "machine_type": machine_type,
        # Recorded so the execute phase reads placement off the receipt rather
        # than a flag: a receipt written before these existed still describes
        # the original region, and the executor falls back to exactly that.
        "zone": zone,
        "region": region,
        "network": c4.NETWORK_NAME,
        "subnetwork": c4.SUBNETWORK_NAME,
        "worker_plan_sha256": plan_sha,
        "worker_plan_job_id": worker_plan["job_id"],
        "samples": worker_plan["samples"],
        "startup_script_relative": STARTUP_RELATIVE,
        "startup_script_sha256": startup_script_sha256(),
        "content_bindings_without_generations": bindings,
        "shards": shards,
        "image_requirement": {
            "family": "debian-12",
            "reason": "engine .so requires GLIBC <= 2.34; debian-12 ships 2.36 "
                      "and python 3.11 matching the cp311 wheelhouse",
        },
        "notes": [
            "read-only planning; no instance, IAM, or write calls exist here",
            "worker service accounts must be designated at execute approval; "
            "the f100 accounts are not used outside their production lifecycle",
        ],
    }
    receipt["plan_receipt_sha256"] = hashlib.sha256(
        canonical_bytes(receipt)
    ).hexdigest()
    return receipt


def dry_run(receipt: Mapping[str, Any]) -> int:
    token = os.environ.get("GOOGLE_OAUTH_ACCESS_TOKEN", "")
    if not token:
        print("dry run needs GOOGLE_OAUTH_ACCESS_TOKEN in the environment; "
              "an intentionally invalid value is acceptable and proves the "
              "requests are constructed and rejected at auth, not before")
        return 2

    from .hu_m31_t3_step6d_fresh_quality_gcp_provider_v1 import GcpQualityRestAdapter

    adapter = GcpQualityRestAdapter(
        access_token=token, zone=receipt.get("zone", DEFAULT_ZONE)
    )
    failures = []
    for binding in receipt["content_bindings_without_generations"]:
        name = binding["object"]
        try:
            found = adapter.get_object_metadata(
                bucket=receipt["bucket"], object_name=name
            )
            state = "absent (stageable)" if found is None else (
                f"present generation {found.get('generation')}"
            )
            print(f"  GET object {name}: {state}")
        except Exception as error:  # noqa: BLE001 - the status line is the result
            print(f"  GET object {name}: {error}")
            failures.append(str(error))
    try:
        machine = adapter.get_machine_type(machine_type=receipt["machine_type"])
        print(f"  GET machineType {receipt['machine_type']}: "
              f"{machine.get('guestCpus')} vCPU")
    except Exception as error:  # noqa: BLE001
        print(f"  GET machineType {receipt['machine_type']}: {error}")
        failures.append(str(error))

    if failures and all("401" in failure for failure in failures):
        print("dry run reached the API surface and stopped at authentication "
              "(HTTP 401) -- request construction is exercised end to end")
        return 0
    if failures:
        print("dry run hit non-auth failures; fix before any execute phase")
        return 1
    print("dry run passed with live credentials; staging names are consistent")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "dryrun"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--worker-plan", required=True)
    parser.add_argument("--out", required=True,
                        help="directory receiving the plan receipt")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--machine-type", default=DEFAULT_MACHINE_TYPE)
    parser.add_argument("--worker-count", type=int, default=DEFAULT_WORKER_COUNT,
                        help="strided workers per instance; match the machine's "
                             "vCPU count when scaling the machine type up")
    parser.add_argument("--zone", default="",
                        help="approved zone to place this run's instances in; "
                             f"defaults to {DEFAULT_ZONE}. Mutually exclusive "
                             "with --region.")
    parser.add_argument("--region", default="",
                        help="approved region to place this run's instances in; "
                             "resolves to that region's approved zone. Defaults "
                             f"to {DEFAULT_ZONE.rsplit('-', 1)[0]}.")
    parser.add_argument("--carry-manifest", default="",
                        help="file of position file names, one per line, that "
                             "an earlier run already published; staged as a "
                             "content binding so every worker skips them")
    args = parser.parse_args()

    if args.zone and args.region:
        raise SystemExit("pass --zone or --region, not both")
    if args.region:
        matches = [z for z in c4.APPROVED_ZONES
                   if z.rsplit("-", 1)[0] == args.region]
        if not matches:
            raise SystemExit(
                f"region {args.region!r} has no approved zone; approved zones "
                f"are {list(c4.APPROVED_ZONES)}"
            )
        zone = matches[0]
    else:
        zone = args.zone or DEFAULT_ZONE

    receipt = build_run_plan(
        run_name=args.run_name,
        package_dir=pathlib.Path(args.package_dir),
        plan_file=pathlib.Path(args.worker_plan),
        bucket=args.bucket,
        machine_type=args.machine_type,
        worker_count=args.worker_count,
        zone=zone,
        carry_manifest=(pathlib.Path(args.carry_manifest)
                        if args.carry_manifest else None),
    )
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    receipt_path = out / f"run_plan_{args.run_name}.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"plan receipt: {receipt_path}")
    print(f"  startup sha256 {receipt['startup_script_sha256'][:16]}...")
    print(f"  {len(receipt['shards'])} shards, "
          f"{sum(s['count'] for s in receipt['shards']):,} positions, "
          f"samples {receipt['samples']}")
    print(f"  zone {receipt['zone']} (region {receipt['region']}), "
          f"subnetwork {receipt['subnetwork']}")
    carried = [b for b in receipt["content_bindings_without_generations"]
               if b["relative"] == CARRY_MANIFEST_RELATIVE]
    if carried:
        print(f"  carry manifest {carried[0]['bytes']:,} bytes "
              f"sha256 {carried[0]['sha256'][:16]}...")
    if args.mode == "plan":
        return 0
    return dry_run(receipt)


if __name__ == "__main__":
    sys.exit(main())
