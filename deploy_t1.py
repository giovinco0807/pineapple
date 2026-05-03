#!/usr/bin/env python3
"""Deploy T1 data generation fleet with SPOT VMs using Rust pipeline.
Each VM generates `hands` * `n1` tasks.
"""
import subprocess
import sys
import os

PROJECT = "ofc-solver-485418"
STARTUP_SCRIPT = r"c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\gcp_t1_startup.sh"

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
]

CONFIG = {
    "prefix": "t1-rust",
    "count": 10,
    "hands": 20,       # 10 VMs * 20 hands = 200 hands total
    "n1": 50,          # 50 deals per hand
    "samples": 10,
    "nesting": "5-2",
    "base_seed": 8000000,
    "seed_step": 10000,
    "gcs_dest": "gs://ofc-solver-485418/t1_mc_data",
    "git_branch": "verify-gcp-phase-one-20260501",
    "machine_type": "e2-highcpu-32"
}

def create_vm(name, zone, worker_id, seed):
    """Create a spot VM via PowerShell."""
    metadata = (
        f"worker-id={worker_id},"
        f"seed={seed},"
        f"hands={CONFIG['hands']},"
        f"n1={CONFIG['n1']},"
        f"samples={CONFIG['samples']},"
        f"nesting={CONFIG['nesting']},"
        f"gcs-dest={CONFIG['gcs_dest']},"
        f"git-branch={CONFIG['git_branch']}"
    )

    ps_cmd = (
        f'gcloud compute instances create {name} '
        f'--project={PROJECT} '
        f'--zone={zone} '
        f'--machine-type={CONFIG["machine_type"]} '
        f'--provisioning-model=SPOT '
        f'--instance-termination-action=DELETE '
        f'--no-restart-on-failure '
        f'--maintenance-policy=TERMINATE '
        f'--image-family=debian-12 '
        f'--image-project=debian-cloud '
        f'--boot-disk-size=30GB '
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
    combined = result.stdout + result.stderr
    has_error = "ERROR" in combined and "WARNING" not in combined.split("ERROR")[0][-50:]
    if result.returncode != 0 and "Created" not in combined:
        err = combined.strip()
        return False, err[:200]
    return True, ""


import concurrent.futures
from threading import Lock

def deploy_instance(i, prefix, count, lock, state):
    name = f"{prefix}-{i}"
    seed = CONFIG["base_seed"] + i * CONFIG["seed_step"]
    zone = ZONES[i % len(ZONES)]

    ok, err = create_vm(name, zone, i, seed)
    if ok:
        with lock:
            state["created"] += 1
            print(f"  [{state['created']:>2}/{count}] {name} in {zone} (seed={seed})")
    else:
        # Retry
        alt_zone = ZONES[(i + 15) % len(ZONES)]
        ok2, err2 = create_vm(name, alt_zone, i, seed)
        if ok2:
            with lock:
                state["created"] += 1
                print(f"  [{state['created']:>2}/{count}] {name} in {alt_zone} (seed={seed}) [fallback]")
        else:
            with lock:
                state["failed"].append((name, err2[:100]))
                print(f"  [FAIL] {name}: {err2[:80]}")

def deploy_fleet():
    prefix = CONFIG["prefix"]
    count = CONFIG["count"]
    print(f"\n{'='*60}")
    print(f"Deploying T1 MC Generation Fleet ({count} VMs, {CONFIG['machine_type']})")
    print(f"  hands/VM={CONFIG['hands']}, n1={CONFIG['n1']}, samples={CONFIG['samples']}")
    print(f"  nesting={CONFIG['nesting']}")
    print(f"  Total hands: {count * CONFIG['hands']}")
    print(f"  GCS: {CONFIG['gcs_dest']}")
    print(f"{'='*60}\n")

    lock = Lock()
    state = {"created": 0, "failed": []}

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(deploy_instance, i, prefix, count, lock, state) for i in range(count)]
        concurrent.futures.wait(futures)

    print(f"\n{'='*60}")
    print(f"SUMMARY: {state['created']}/{count} VMs created. {len(state['failed'])} failed.")
    if state["failed"]:
        for name, err in state["failed"]:
            print(f"  {name}: {err}")
    print(f"{'='*60}")

if __name__ == "__main__":
    deploy_fleet()
