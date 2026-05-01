#!/usr/bin/env python3
"""Delete all t0-* GCP VMs."""
import subprocess, json

result = subprocess.run(
    ["gcloud", "compute", "instances", "list", "--filter=name~t0-", "--format=json"],
    capture_output=True, text=True
)
vms = json.loads(result.stdout)
print(f"Found {len(vms)} VMs to delete")

for vm in vms:
    name = vm["name"]
    zone = vm["zone"].split("/")[-1]
    print(f"Deleting {name} in {zone}...")
    subprocess.run(
        ["gcloud", "compute", "instances", "delete", name, f"--zone={zone}", "--quiet"],
        capture_output=True, text=True
    )
    print(f"  Done: {name}")

print(f"\nAll {len(vms)} VMs deleted.")
