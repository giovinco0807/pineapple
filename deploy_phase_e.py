#!/usr/bin/env python3
"""Deploy Phase E fleet with SPOT VMs.
Uses NN-filtered JSON slices from GCS.
Each VM processes its own slice of 100 hands.
"""
import subprocess
import sys

PROJECT = "ofc-solver-485418"
STARTUP_SCRIPT = r"c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\gcp_worker_startup_v4_nn.sh"

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
]

CONFIG = {
    "prefix": "t0-e",
    "count": 20,
    "samples": 30,
    "nesting": "3;2;2",  # semicolons to avoid metadata comma issues
    "base_seed": 5000000,
    "seed_step": 100000,
    "gcs_dest": "gs://ofc-solver-485418/t0_phase_e",
    "git_branch": "verify-gcp-phase-one-20260501",
}


def create_vm(name, zone, worker_id, seed, gcs_input):
    """Create a spot VM via PowerShell."""
    metadata = (
        f"worker-id={worker_id},"
        f"seed={seed},"
        f"samples={CONFIG['samples']},"
        f"nesting={CONFIG['nesting']},"
        f"gcs-input={gcs_input},"
        f"gcs-dest={CONFIG['gcs_dest']},"
        f"git-branch={CONFIG['git_branch']}"
    )

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
    combined = result.stdout + result.stderr
    # Only treat as failure if actual ERROR (not WARNING) or non-zero exit
    has_error = "ERROR" in combined and "WARNING" not in combined.split("ERROR")[0][-50:]
    if result.returncode != 0 and "Created" not in combined:
        err = combined.strip()
        return False, err[:200]
    return True, ""


def deploy_fleet():
    prefix = CONFIG["prefix"]
    count = CONFIG["count"]
    print(f"\n{'='*60}")
    print(f"Phase E: NN-Filtered T0 Fleet ({count} VMs)")
    print(f"  samples={CONFIG['samples']}, nesting={CONFIG['nesting']}")
    print(f"  100 hands/VM × {count} VMs = 2000 hands total")
    print(f"  GCS: {CONFIG['gcs_dest']}")
    print(f"  Provisioning: SPOT")
    print(f"{'='*60}\n")

    created = 0
    failed = []

    for i in range(count):
        name = f"{prefix}-{i}"
        seed = CONFIG["base_seed"] + i * CONFIG["seed_step"]
        gcs_input = f"{CONFIG['gcs_dest']}/slices/phase_e_slice_{i:02d}.json"
        zone = ZONES[i % len(ZONES)]

        ok, err = create_vm(name, zone, i, seed, gcs_input)
        if ok:
            created += 1
            print(f"  [{created:>2}/{count}] {name} in {zone} (seed={seed})")
        else:
            # Retry with alternate zone
            alt_zone = ZONES[(i + 15) % len(ZONES)]
            ok2, err2 = create_vm(name, alt_zone, i, seed, gcs_input)
            if ok2:
                created += 1
                print(f"  [{created:>2}/{count}] {name} in {alt_zone} (seed={seed}) [fallback]")
            else:
                failed.append((name, err2[:100]))
                print(f"  [FAIL] {name}: {err2[:80]}")

    print(f"\n{'='*60}")
    print(f"SUMMARY: {created}/{count} VMs created. {len(failed)} failed.")
    if failed:
        print("Failed VMs:")
        for name, err in failed:
            print(f"  {name}: {err}")
    print(f"{'='*60}")


if __name__ == "__main__":
    deploy_fleet()
