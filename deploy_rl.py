#!/usr/bin/env python3
"""Deploy Continuous RL on a single GCP SPOT VM."""
import subprocess
import os

PROJECT = "ofc-solver-485418"
STARTUP_SCRIPT = r"c:\Users\Owner\.gemini\antigravity\worktrees\ofc-pineapple\verify-gcp-phase-one-20260501\gcp_rl_startup.sh"

CONFIG = {
    "name": "rl-worker-1",
    "zone": "us-central1-a",
    "git_branch": "verify-gcp-phase-one-20260501",
    "machine_type": "c2-standard-60" # High CPU for MCTS
}

def create_vm():
    """Create a spot VM via PowerShell."""
    metadata = f"git-branch={CONFIG['git_branch']}"

    ps_cmd = (
        f'gcloud compute instances create {CONFIG["name"]} '
        f'--project={PROJECT} '
        f'--zone={CONFIG["zone"]} '
        f'--machine-type={CONFIG["machine_type"]} '
        f'--provisioning-model=SPOT '
        f'--instance-termination-action=DELETE '
        f'--no-restart-on-failure '
        f'--maintenance-policy=TERMINATE '
        f'--image-family=debian-12 '
        f'--image-project=debian-cloud '
        f'--boot-disk-size=50GB '
        f'--boot-disk-type=pd-standard '
        f'--scopes=storage-full '
        f'"--metadata={metadata}" '
        f'"--metadata-from-file=startup-script={STARTUP_SCRIPT}" '
    )

    print(f"Deploying {CONFIG['name']} in {CONFIG['zone']}...")
    result = subprocess.run(
        ["powershell", "-Command", ps_cmd],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

if __name__ == "__main__":
    create_vm()
