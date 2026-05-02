#!/usr/bin/env python3
"""Deploy remaining Phase E VMs (7-10, 12-19)."""
import subprocess

PROJECT = "ofc-solver-485418"
STARTUP_SCRIPT = r"c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\gcp_worker_startup_v4_nn.sh"

ZONES = [
    "us-east4-a", "us-east4-b", "us-east4-c",
    "us-west1-a", "us-west1-b",
    "us-west4-a", "us-west4-b",
    "us-south1-a",
    "northamerica-northeast1-a", "northamerica-northeast1-b",
    "europe-west1-b", "europe-west1-c",
    "europe-west2-a", "europe-west3-a",
    "europe-west4-a", "asia-east1-a",
    "asia-northeast1-a", "asia-southeast1-a",
    "southamerica-east1-a", "australia-southeast1-a",
]

CONFIG = {
    "samples": 30,
    "nesting": "3;2;2",
    "base_seed": 5000000,
    "seed_step": 100000,
    "gcs_dest": "gs://ofc-solver-485418/t0_phase_e",
    "git_branch": "verify-gcp-phase-one-20260501",
}


def create_vm(name, zone, worker_id, seed, gcs_input):
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
    # "Created" means success even with WARNING
    if "Created" in combined:
        return True, ""
    err = combined.strip()
    return False, err[:200]


if __name__ == "__main__":
    # VMs 7-10 and 12-19 (VM 11 already created)
    remaining = [7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19]
    print(f"Deploying {len(remaining)} remaining Phase E VMs...")

    created = 0
    for idx, i in enumerate(remaining):
        name = f"t0-e-{i}"
        seed = CONFIG["base_seed"] + i * CONFIG["seed_step"]
        gcs_input = f"{CONFIG['gcs_dest']}/slices/phase_e_slice_{i:02d}.json"
        zone = ZONES[idx % len(ZONES)]

        ok, err = create_vm(name, zone, i, seed, gcs_input)
        if ok:
            created += 1
            print(f"  [{created:>2}/{len(remaining)}] {name} in {zone} (seed={seed})")
        else:
            print(f"  [FAIL] {name}: {err[:120]}")

    print(f"\nDone: {created}/{len(remaining)} VMs created.")
