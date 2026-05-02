#!/usr/bin/env python3
"""
Deploy T1 MC data generation to GCP Spot VM.

Creates a spot VM, uploads files, runs generation, downloads results.

Usage:
    python ai/deploy_t1_gcp.py
"""

import subprocess
import sys
import time
import os

# ── Configuration ──
PROJECT = "ofc-solver-485418"
ZONE = "us-central1-a"
VM_NAME = "t1-mc-gen"
MACHINE_TYPE = "e2-highcpu-32"  # 32 vCPUs, spot
REMOTE_USER = None  # auto-detect
REMOTE_DIR = "/tmp/t1_gen"
LOCAL_AI_DIR = os.path.dirname(os.path.abspath(__file__))


def run(cmd, check=True, capture=False):
    print(f"  $ {cmd}")
    if capture:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if check and r.returncode != 0:
            print(f"    STDERR: {r.stderr}")
            raise RuntimeError(f"Command failed: {cmd}")
        return r.stdout.strip()
    else:
        r = subprocess.run(cmd, shell=True)
        if check and r.returncode != 0:
            raise RuntimeError(f"Command failed: {cmd}")
        return ""


def create_vm():
    print(f"\n=== Creating or Starting Spot VM: {VM_NAME} ({MACHINE_TYPE}) ===")
    res = run(f"gcloud compute instances describe {VM_NAME} --zone={ZONE} --project={PROJECT} --format=json", check=False, capture=True)
    if "ERROR" not in res and VM_NAME in res:
        print("  VM already exists. Starting it...")
        run(f"gcloud compute instances start {VM_NAME} --zone={ZONE} --project={PROJECT}")
    else:
        run(f'gcloud compute instances create {VM_NAME} '
            f'--project={PROJECT} '
            f'--zone={ZONE} '
            f'--machine-type={MACHINE_TYPE} '
            f'--provisioning-model=SPOT '
            f'--instance-termination-action=STOP '
            f'--image-family=debian-12 '
            f'--image-project=debian-cloud '
            f'--boot-disk-size=20GB '
            f'--boot-disk-type=pd-ssd '
            f'--no-restart-on-failure ')

    print("  Waiting for VM to start...")
    time.sleep(15)

    # Wait for SSH to be ready
    for attempt in range(10):
        try:
            run(f'gcloud compute ssh {VM_NAME} --zone={ZONE} --command="echo ready" --quiet', check=True, capture=True)
            print("  VM is ready!")
            return
        except:
            print(f"  SSH not ready yet (attempt {attempt+1}/10)...")
            time.sleep(10)
    raise RuntimeError("VM failed to become SSH-ready")


def get_remote_user():
    global REMOTE_USER
    if REMOTE_USER is None:
        REMOTE_USER = ssh("whoami", capture=True).strip()
        print(f"  Remote user: {REMOTE_USER}")
    return REMOTE_USER


def upload_files():
    print(f"\n=== Uploading files to VM ===")
    # Create remote directory
    ssh(f"mkdir -p {REMOTE_DIR}")

    # Package rust_solver locally first
    print("  Packaging rust_solver...")
    run(f'cd "{LOCAL_AI_DIR}" && tar -czf rust_solver.tar.gz rust_solver')

    # Files to upload
    files = [
        os.path.join(LOCAL_AI_DIR, "rust_solver.tar.gz"),
        os.path.join(LOCAL_AI_DIR, "gcp_t1_setup.sh"),
    ]

    for f in files:
        fname = os.path.basename(f)
        print(f"  Uploading {fname}...")
        run(f'gcloud compute scp "{f}" {VM_NAME}:{REMOTE_DIR}/{fname} --zone={ZONE} --quiet')




def ssh(cmd, check=True, capture=False):
    return run(f'gcloud compute ssh {VM_NAME} --zone={ZONE} --command="{cmd}" --quiet', check=check, capture=capture)


def setup_and_run():
    print(f"\n=== Setting up environment and starting generation ===")
    # Run setup in tmux so it survives disconnection
    ssh(f"chmod +x {REMOTE_DIR}/gcp_t1_setup.sh")

    # Install tmux
    ssh("sudo apt-get update -qq && sudo apt-get install -y -qq tmux", check=False)

    # Start generation in tmux
    ssh(f"tmux new-session -d -s t1gen 'cd {REMOTE_DIR} && bash gcp_t1_setup.sh 2>&1 | tee run.log'")

    print("\n  Generation started in tmux session 't1gen'!")
    print(f"  Monitor with: gcloud compute ssh {VM_NAME} --zone={ZONE} --command=\"tmux attach -t t1gen\"")
    print(f"  Or check logs: gcloud compute ssh {VM_NAME} --zone={ZONE} --command=\"tail -f ~/t1_gen/run.log\"")


def download_results():
    print(f"\n=== Downloading results ===")
    local_out = os.path.join(LOCAL_AI_DIR, "data", "t1_data_50k.jsonl")
    run(f'gcloud compute scp {VM_NAME}:{REMOTE_DIR}/rust_solver/t1_data_50k.jsonl "{local_out}" --zone={ZONE} --quiet')
    print(f"  Downloaded to: {local_out}")


def check_status():
    print(f"\n=== Checking generation status ===")
    output = ssh(f"tail -5 {REMOTE_DIR}/run.log 2>/dev/null || echo 'No log yet'", capture=True)
    print(output)

    # Check if generation is complete
    done = ssh(f"grep -c 'All done' {REMOTE_DIR}/run.log 2>/dev/null || echo 0", capture=True)
    return done.strip() != "0"


def delete_vm():
    print(f"\n=== Deleting VM: {VM_NAME} ===")
    run(f'gcloud compute instances delete {VM_NAME} --zone={ZONE} --quiet', check=False)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Deploy T1 MC generation to GCP")
    parser.add_argument('action', choices=['deploy', 'status', 'download', 'delete', 'all'],
                       help="Action to perform")
    args = parser.parse_args()

    if args.action == 'deploy':
        create_vm()
        upload_files()
        setup_and_run()
        print("\n✓ Deployment complete! Use 'status' to check progress.")

    elif args.action == 'status':
        check_status()

    elif args.action == 'download':
        download_results()

    elif args.action == 'delete':
        delete_vm()

    elif args.action == 'all':
        create_vm()
        upload_files()
        setup_and_run()

        print("\n=== Waiting for completion (checking every 5 min) ===")
        while True:
            time.sleep(300)
            try:
                if check_status():
                    break
            except:
                print("  (check failed, retrying...)")

        download_results()
        delete_vm()
        print("\n✓ All done!")


if __name__ == '__main__':
    main()
