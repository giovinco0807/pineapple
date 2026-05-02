"""
Phase F: Deploy solver to all GCP instances using script upload approach.
1. Upload start_solver.sh to each instance
2. Run it with the shard ID
"""
import subprocess
import sys
import time

INSTANCES = [
    ("t0-e-0",  "us-central1-a"),
    ("t0-e-2",  "us-central1-c"),
    ("t0-e-3",  "us-central1-f"),
    ("t0-e-18", "europe-west1-b"),
    ("t0-e-19", "europe-west1-c"),
    ("t0-e-3",  "europe-west1-c"),
    ("t0-e-10", "us-west1-a"),
    ("t0-e-12", "us-west1-b"),
    ("t0-e-4",  "us-east1-b"),
    ("t0-e-5",  "us-east1-c"),
    ("t0-e-6",  "us-east1-d"),
    ("t0-e-7",  "us-east4-a"),
    ("t0-e-8",  "us-east4-b"),
    ("t0-e-9",  "us-east4-c"),
    ("t0-e-26", "australia-southeast1-a"),
    ("t0-e-20", "europe-west2-b"),
    ("t0-e-21", "europe-west3-a"),
    ("t0-e-25", "southamerica-east1-a"),
    ("t0-e-0",  "northamerica-northeast1-a"),
    ("t0-e-16", "northamerica-northeast1-a"),
    ("t0-e-17", "northamerica-northeast1-b"),
    ("t0-e-22", "europe-west4-a"),
    ("t0-e-13", "us-west4-a"),
    ("t0-e-14", "us-west4-b"),
    ("t0-e-15", "us-south1-a"),
    ("t0-e-28", "me-west1-a"),
    ("t0-e-29", "africa-south1-a"),
]

SCRIPT = "start_solver.sh"


def ps(cmd, timeout=300):
    r = subprocess.run(["powershell", "-Command", cmd],
                       capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip(), r.stderr.strip(), r.returncode


def deploy(idx, name, zone):
    shard_id = f"{idx:02d}"
    print(f"\n[{idx+1}/{len(INSTANCES)}] {name} ({zone}) shard={shard_id}")
    
    # Upload script
    out, err, rc = ps(f"gcloud compute scp {SCRIPT} {name}:/tmp/{SCRIPT} --zone={zone}", timeout=60)
    if rc != 0:
        print(f"  [FAIL] scp: {err[:200]}")
        return False
    
    # Run script as root
    out, err, rc = ps(
        f"gcloud compute ssh {name} --zone={zone} --command=\"sudo bash /tmp/{SCRIPT} {shard_id}\"",
        timeout=600
    )
    
    if "Setup Complete" in out or "Solver PID" in out:
        print(f"  [OK] {out.split(chr(10))[-2] if out else 'started'}")
        return True
    else:
        print(f"  [WARN] RC={rc}")
        # Show last few lines
        for line in out.split('\n')[-5:]:
            if line.strip():
                print(f"  > {line.strip()}")
        if err:
            print(f"  err: {err[-200:]}")
        return rc == 0


def main():
    print("=" * 60)
    print("Phase F: Deploy solver via script upload")
    print("=" * 60)
    
    ok = 0
    for i, (name, zone) in enumerate(INSTANCES):
        try:
            if deploy(i, name, zone):
                ok += 1
        except Exception as e:
            print(f"  [EXCEPTION] {e}")
        time.sleep(1)
    
    print(f"\n{'='*60}")
    print(f"Started: {ok}/{len(INSTANCES)}")


if __name__ == "__main__":
    main()
