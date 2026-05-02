"""
Monitor progress of T0 filtered v4 evaluation across GCP instances.
Shows per-instance progress and estimates completion time.
"""
import subprocess
import sys

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


def ssh_cmd(name, zone, cmd):
    full_cmd = ["gcloud", "compute", "ssh", name, f"--zone={zone}", f"--command={cmd}"]
    try:
        result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=30)
        return result.stdout.strip(), result.returncode
    except:
        return "", 1


def main():
    total_done = 0
    total_expected = 0
    running = 0
    finished = 0
    
    print(f"{'Idx':>3} {'Instance':<10} {'Zone':<30} {'Status':<10} {'Done':>5} {'Log tail'}")
    print("-" * 100)
    
    for i, (name, zone) in enumerate(INSTANCES):
        # Check if process is running
        ps_out, _ = ssh_cmd(name, zone, "pgrep -c cfr_solver 2>/dev/null || echo 0")
        is_running = ps_out.strip() != "0" and ps_out.strip() != ""
        
        # Count output lines
        out_file = f"t0_filtered_output_{i}.jsonl"
        wc_out, rc = ssh_cmd(name, zone, f"wc -l ~/pineapple/{out_file} 2>/dev/null || echo 0")
        n_done = 0
        try:
            n_done = int(wc_out.split()[0])
        except:
            pass
        
        # Get last log line
        log_out, _ = ssh_cmd(name, zone, f"tail -1 ~/pineapple/solver_{i}.log 2>/dev/null || echo 'no log'")
        log_tail = log_out[:60] if log_out else "no log"
        
        status = "RUNNING" if is_running else ("DONE" if n_done > 0 else "IDLE")
        if status == "RUNNING":
            running += 1
        elif status == "DONE":
            finished += 1
        
        total_done += n_done
        
        print(f"{i:>3} {name:<10} {zone:<30} {status:<10} {n_done:>5} {log_tail}")
    
    print("-" * 100)
    print(f"Total hands completed: {total_done}")
    print(f"Running: {running}, Finished: {finished}, Idle: {len(INSTANCES)-running-finished}")


if __name__ == "__main__":
    main()
