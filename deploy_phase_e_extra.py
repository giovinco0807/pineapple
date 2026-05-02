#!/usr/bin/env python3
"""Deploy t0-e-6 (redeploy) + t0-e-20 to t0-e-29 (10 new VMs)."""
import subprocess

PROJECT = "ofc-solver-485418"
STARTUP_SCRIPT = r"c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\gcp_worker_startup_v4_nn.sh"

CONFIG = {
    "samples": 30,
    "nesting": "3;2;2",
    "gcs_dest": "gs://ofc-solver-485418/t0_phase_e",
    "git_branch": "verify-gcp-phase-one-20260501",
}

# VM definitions: (name, zone, worker_id, seed)
VMS = [
    # Redeploy
    ("t0-e-6",  "us-east1-d",              6,  5600000),
    # New VMs 20-29
    ("t0-e-20", "europe-west2-a",          20,  7100000),
    ("t0-e-21", "europe-west3-a",          21,  7200000),
    ("t0-e-22", "europe-west4-a",          22,  7300000),
    ("t0-e-23", "asia-northeast1-a",       23,  7400000),
    ("t0-e-24", "asia-southeast1-a",       24,  7500000),  
    ("t0-e-25", "southamerica-east1-a",    25,  7600000),
    ("t0-e-26", "australia-southeast1-a",  26,  7700000),
    ("t0-e-27", "europe-north1-a",         27,  7800000),
    ("t0-e-28", "me-west1-a",             28,  7900000),
    ("t0-e-29", "africa-south1-a",         29,  8000000),
]


def create_vm(name, zone, worker_id, seed):
    gcs_input = f"{CONFIG['gcs_dest']}/slices/phase_e_slice_{worker_id:02d}.json"
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
    if "Created" in combined:
        return True, ""
    err = combined.strip()
    return False, err[:200]


if __name__ == "__main__":
    print(f"Deploying {len(VMS)} VMs (1 redeploy + 10 new)...")
    created = 0
    for name, zone, wid, seed in VMS:
        ok, err = create_vm(name, zone, wid, seed)
        if ok:
            created += 1
            print(f"  [{created:>2}/{len(VMS)}] {name} in {zone} (worker={wid}, seed={seed})")
        else:
            print(f"  [FAIL] {name}: {err[:120]}")
    print(f"\nDone: {created}/{len(VMS)} VMs created.")
