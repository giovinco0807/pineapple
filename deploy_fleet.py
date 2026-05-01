#!/usr/bin/env python3
"""Deploy Phase D fleet with SPOT VMs.
Handles nesting comma issue by using semicolons in metadata,
which the startup script converts back to commas.
"""
import subprocess
import json
import sys
import os
import tempfile

PROJECT = "ofc-solver-485418"
STARTUP_SCRIPT = r"c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\gcp_worker_startup_v3.sh"

ZONES = [
    "us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f",
    "us-east1-b", "us-east1-c", "us-east1-d",
    "us-east4-a", "us-east4-b", "us-east4-c",
    "us-west1-a", "us-west1-b",
    "us-west4-a", "us-west4-b",
    "us-south1-a",
    "northamerica-northeast1-a", "northamerica-northeast1-b",
    "europe-west1-b", "europe-west1-c",
    "europe-west2-a",
    "europe-west3-a",
    "europe-west4-a",
    "asia-east1-a",
    "asia-northeast1-a",
    "asia-southeast1-a",
    "southamerica-east1-a",
    "australia-southeast1-a",
    "me-west1-a",
    "africa-south1-a",
    "europe-north1-a",
    "europe-west6-a",
    "asia-south1-a",
]

TRAIN_CONFIG = {
    "prefix": "t0-d",
    "count": 30,
    "hands": 17,
    "samples": 30,
    "nesting": "5;3;2",  # semicolons to avoid metadata comma issues
    "top_k": 100,
    "base_seed": 3000000,
    "seed_step": 50000,
    "gcs_dest": "gs://ofc-solver-485418/t0_phase_d",
}

TEST_CONFIG = {
    "prefix": "t0-test",
    "count": 2,
    "hands": 25,
    "samples": 30,
    "nesting": "10;3;2",
    "top_k": 0,
    "base_seed": 6000000,
    "seed_step": 50000,
    "gcs_dest": "gs://ofc-solver-485418/t0_test_gold",
}


def create_vm(name, zone, worker_id, seed, hands, samples, nesting, top_k, gcs_dest):
    """Create a spot VM via PowerShell."""
    metadata = (
        f"worker-id={worker_id},"
        f"seed={seed},"
        f"hands={hands},"
        f"samples={samples},"
        f"nesting={nesting},"
        f"top-k={top_k},"
        f"gcs-dest={gcs_dest}"
    )

    # PowerShell command - properly quoted
    ps_cmd = (
        f'gcloud compute instances create {name} '
        f'--project={PROJECT} '
        f'--zone={zone} '
        f'--machine-type=e2-highcpu-16 '
        f'--provisioning-model=SPOT '
        f'--instance-termination-action=DELETE '
        f'--no-restart-on-failure '
        f'--maintenance-policy=TERMINATE '
        f'--image-family=debian-12 '
        f'--image-project=debian-cloud '
        f'--boot-disk-size=20GB '
        f'--boot-disk-type=pd-standard '
        f'--scopes=storage-full '
        f'"--metadata={metadata}" '
        f'"--metadata-from-file=startup-script={STARTUP_SCRIPT}" '
        f'--quiet 2>&1'
    )

    result = subprocess.run(
        ["powershell", "-Command", ps_cmd],
        capture_output=True, text=True
    )
    if result.returncode != 0 or "ERROR" in result.stderr:
        err = (result.stderr or result.stdout).strip()
        return False, err[:200]
    return True, ""


def deploy_fleet(config, start_zone_idx=0):
    prefix = config["prefix"]
    count = config["count"]
    print(f"\n{'='*60}")
    print(f"Deploying {count} VMs: {prefix}-0 to {prefix}-{count-1}")
    print(f"  hands={config['hands']}, samples={config['samples']}")
    print(f"  nesting={config['nesting']}, top-k={config['top_k']}")
    print(f"  GCS: {config['gcs_dest']}")
    print(f"  Provisioning: SPOT")
    print(f"{'='*60}\n")

    created = 0
    failed = []

    for i in range(count):
        name = f"{prefix}-{i}"
        seed = config["base_seed"] + i * config["seed_step"]
        zone = ZONES[(start_zone_idx + i) % len(ZONES)]

        ok, err = create_vm(
            name, zone, i, seed,
            config["hands"], config["samples"],
            config["nesting"], config["top_k"], config["gcs_dest"]
        )
        if ok:
            created += 1
            print(f"  [{created:>2}/{count}] {name} in {zone} (seed={seed})")
        else:
            alt_zone = ZONES[(start_zone_idx + i + 15) % len(ZONES)]
            ok2, err2 = create_vm(
                name, alt_zone, i, seed,
                config["hands"], config["samples"],
                config["nesting"], config["top_k"], config["gcs_dest"]
            )
            if ok2:
                created += 1
                print(f"  [{created:>2}/{count}] {name} in {alt_zone} (seed={seed}) [fallback]")
            else:
                failed.append((name, err2[:100]))
                print(f"  [FAIL] {name}: {err2[:80]}")

    print(f"\n{prefix}: {created}/{count} VMs created. {len(failed)} failed.")
    return created, failed


if __name__ == "__main__":
    print("=" * 60)
    print("Phase D Fleet Deployment (SPOT VMs, top-K=100)")
    print("=" * 60)

    t_created, t_failed = deploy_fleet(TRAIN_CONFIG, start_zone_idx=0)
    g_created, g_failed = deploy_fleet(TEST_CONFIG, start_zone_idx=5)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"  Training: {t_created}/30 VMs")
    print(f"  Test:     {g_created}/2 VMs")
    print(f"  Total:    {t_created + g_created}/32 VMs")
    print(f"{'='*60}")
