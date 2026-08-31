"""
Optuna GCP parallel tuning v2: bust_penalty × fl_ev_scale → maximize score.

Each trial runs 1000 games using 16 parallel processes on a single VM (~8 min/trial).
Uses manual ask/tell with as_completed for robust parallel scheduling.

Usage:
    python ai/auto_tune_gcp_v2.py --trials 44 --games 1000
    python ai/auto_tune_gcp_v2.py --trials 44 --games 1000 --resume
"""
import argparse
import subprocess
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import optuna

# GCP VM pool (16 VMs)
VM_POOL = [
    ("ofc-selfplay-3", "us-west1-b"),
    ("ofc-selfplay-8", "us-east1-d"),
    ("ofc-selfplay-10", "us-west1-c"),
    ("ofc-selfplay-11", "europe-west1-c"),
    ("ofc-selfplay-14", "us-east4-c"),
    ("ofc-selfplay-17", "us-east1-b"),
    ("ofc-selfplay-24", "us-east4-c"),
    ("ofc-selfplay-25", "us-west4-a"),
    ("ofc-selfplay-26", "us-central1-a"),
    ("ofc-selfplay-27", "us-central1-b"),
    ("ofc-selfplay-28", "europe-west4-a"),
    ("ofc-selfplay-29", "us-west2-a"),
    ("ofc-selfplay-30", "us-east1-c"),
    ("ofc-selfplay-31", "us-west3-a"),
    ("ofc-selfplay-32", "northamerica-northeast1-a"),
    ("ofc-selfplay-33", "us-south1-a"),
]

# Thread-safe VM allocation with blacklist
vm_lock = threading.Lock()
vm_available = []
vm_blacklist = set()  # VM indices that have failed SSH
vm_fail_count = {}    # vm_idx -> consecutive failure count
vm_event = threading.Event()

MAX_CONSECUTIVE_FAILS = 2  # Blacklist after this many consecutive failures


def acquire_vm():
    """Get a free VM index, blocking until one is available."""
    while True:
        with vm_lock:
            if vm_available:
                return vm_available.pop(0)
        vm_event.wait(timeout=5)
        vm_event.clear()


def release_vm(idx, failed=False):
    """Return a VM to the pool. If failed, track and possibly blacklist."""
    with vm_lock:
        if failed:
            vm_fail_count[idx] = vm_fail_count.get(idx, 0) + 1
            if vm_fail_count[idx] >= MAX_CONSECUTIVE_FAILS:
                vm_blacklist.add(idx)
                vm_name = VM_POOL[idx][0]
                print(f"  *** BLACKLISTED {vm_name} after {vm_fail_count[idx]} consecutive failures ***",
                      flush=True)
                return  # Don't put back in pool
        else:
            vm_fail_count[idx] = 0  # Reset on success
        vm_available.append(idx)
    vm_event.set()


def _ssh(vm_name, zone, remote_cmd, timeout=60):
    """Run a short SSH command. Returns (stdout+stderr, success)."""
    cmd = f'gcloud compute ssh {vm_name} --zone={zone} --command="{remote_cmd}"'
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, encoding='utf-8', errors='replace'
        )
        return result.stdout + result.stderr, result.returncode == 0
    except subprocess.TimeoutExpired:
        return "", False
    except Exception as e:
        return str(e), False


def run_benchmark_on_vm(vm_idx, bust_penalty, fl_ev_scale, games, seed):
    """Launch benchmark in background on VM, poll for results (avoids SSH timeout)."""
    vm_name, zone = VM_POOL[vm_idx]
    outfile = f"/tmp/bench_{seed}.txt"

    # Step 1: Launch benchmark in background via nohup (unique output file per trial)
    launch_cmd = (
        f"nohup bash /home/Owner/rust_benchmark/remote_bench.sh "
        f"{bust_penalty:.2f} {fl_ev_scale:.2f} {games} {seed} "
        f"> {outfile} 2>&1 & echo LAUNCHED"
    )
    output, ok = _ssh(vm_name, zone, launch_cmd, timeout=30)
    if "LAUNCHED" not in output:
        print(f"  [{vm_name}] Failed to launch benchmark")
        return None

    # Step 2: Poll for RESULT line (short SSH every 30s, max 10h)
    max_polls = 1200  # 1200 * 30s = 10 hours
    for i in range(max_polls):
        time.sleep(30)
        poll_cmd = f"grep '^RESULT ' {outfile} 2>/dev/null || echo PENDING"
        output, ok = _ssh(vm_name, zone, poll_cmd, timeout=30)

        if not ok:
            continue

        for line in output.splitlines():
            if line.startswith("RESULT ") and "score=" in line:
                m = re.search(
                    r'score=([+-]?\d+\.?\d*)\s+bust=([+-]?\d+\.?\d*)\s+'
                    r'fl=([+-]?\d+\.?\d*)\s+win=([+-]?\d+\.?\d*)',
                    line
                )
                if m:
                    return {
                        "score": float(m.group(1)),
                        "bust": float(m.group(2)),
                        "fl": float(m.group(3)),
                        "win": float(m.group(4)),
                    }
            if line.startswith("RESULT FAILED"):
                print(f"  [{vm_name}] Benchmark returned FAILED")
                return None

    print(f"  [{vm_name}] Timeout (10h polling)")
    return None


def main():
    parser = argparse.ArgumentParser(description="Optuna GCP parallel tuning v2")
    parser.add_argument("--trials", type=int, default=44)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=300000)
    parser.add_argument("--n-vms", type=int, default=8)
    parser.add_argument("--study-name", default="ofc_tune_v2")
    parser.add_argument("--db", default="sqlite:///ai/auto_tune_results.db")
    parser.add_argument("--resume", action="store_true",
                        help="Resume existing study")
    args = parser.parse_args()

    # Limit VM pool
    global vm_available
    pool_size = min(args.n_vms, len(VM_POOL))
    vm_available = list(range(pool_size))

    print(f"=== Optuna GCP Tuning v2 ===")
    print(f"  Trials: {args.trials}")
    print(f"  Games/trial: {args.games} (single process, all cores)")
    print(f"  VMs: {pool_size}")
    print(f"  Params: bust_penalty [0, 20], fl_ev_scale [0.3, 3.0]")
    print(f"  Est. time: {(args.trials / pool_size) * 10:.0f} min")
    print()

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.db,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    if not args.resume:
        study.enqueue_trial({"bust_penalty": 0.0, "fl_ev_scale": 1.0})   # baseline
        study.enqueue_trial({"bust_penalty": 5.0, "fl_ev_scale": 2.0})   # v1 promising
        study.enqueue_trial({"bust_penalty": 10.0, "fl_ev_scale": 1.5})  # mid range
        study.enqueue_trial({"bust_penalty": 17.0, "fl_ev_scale": 2.0})  # v1 best

    # Parameter distributions for ask/tell API
    distributions = {
        "bust_penalty": optuna.distributions.FloatDistribution(0.0, 20.0),
        "fl_ev_scale": optuna.distributions.FloatDistribution(0.3, 3.0),
    }

    def run_trial(trial_obj, games, base_seed):
        bust_penalty = trial_obj.params["bust_penalty"]
        fl_ev_scale = trial_obj.params["fl_ev_scale"]
        vm_idx = acquire_vm()
        vm_name = VM_POOL[vm_idx][0]
        seed = base_seed + trial_obj.number * 10000
        try:
            print(f"  Trial {trial_obj.number}: bp={bust_penalty:.1f} "
                  f"fl={fl_ev_scale:.2f} -> {vm_name} ({games}g)", flush=True)
            metrics = run_benchmark_on_vm(
                vm_idx, bust_penalty, fl_ev_scale, games, seed
            )
            if metrics is None:
                release_vm(vm_idx, failed=True)
                return trial_obj, float("-inf"), {}
            release_vm(vm_idx, failed=False)
            return trial_obj, metrics["score"], metrics
        except Exception:
            release_vm(vm_idx, failed=True)
            return trial_obj, float("-inf"), {}

    completed_count = 0
    valid_count = 0
    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        futures = {}
        # Submit initial batch
        for _ in range(min(args.trials, pool_size)):
            trial = study.ask(distributions)
            f = executor.submit(run_trial, trial, args.games, args.seed)
            futures[f] = trial

        submitted = len(futures)

        for f in as_completed(futures):
            trial_obj, score, metrics = f.result()
            if metrics:
                trial_obj.set_user_attr("bust_rate", metrics.get("bust", -1))
                trial_obj.set_user_attr("fl_rate", metrics.get("fl", -1))
                trial_obj.set_user_attr("win_rate", metrics.get("win", -1))
            study.tell(trial_obj, score)
            completed_count += 1

            if score > float("-inf"):
                valid_count += 1
                label = f"score={score:+.2f}"
            else:
                label = "FAILED"
            bp = trial_obj.params["bust_penalty"]
            fl_s = trial_obj.params["fl_ev_scale"]
            print(f"  [{completed_count}/{args.trials}] Trial {trial_obj.number}: {label} "
                  f"bp={bp:.1f} fl_s={fl_s:.2f} ({valid_count} valid)", flush=True)
            if metrics:
                print(f"    bust={metrics.get('bust','?')}% fl={metrics.get('fl','?')}% "
                      f"win={metrics.get('win','?')}%", flush=True)

            # Submit next trial if available and we still have working VMs
            if submitted < args.trials:
                with vm_lock:
                    active_vms = pool_size - len(vm_blacklist)
                if active_vms > 0:
                    trial = study.ask(distributions)
                    f2 = executor.submit(run_trial, trial, args.games, args.seed)
                    futures[f2] = trial
                    submitted += 1
                else:
                    print("  *** All VMs blacklisted! Stopping. ***", flush=True)
                    break

    # Print results
    print(f"\n{'='*60}")
    print(f"  OPTUNA RESULTS ({len(study.trials)} trials, {args.games} games/trial)")
    print(f"{'='*60}")

    valid_trials = [t for t in study.trials
                    if t.value is not None and t.value > float("-inf")]

    if not valid_trials:
        print("  No valid trials!")
        return

    best = study.best_trial
    print(f"  Best score: {best.value:+.2f}")
    print(f"  bust_penalty: {best.params['bust_penalty']:.2f}")
    print(f"  fl_ev_scale:  {best.params['fl_ev_scale']:.2f}")
    print(f"  bust_rate: {best.user_attrs.get('bust_rate', '?')}%")
    print(f"  fl_rate:   {best.user_attrs.get('fl_rate', '?')}%")
    print(f"  win_rate:  {best.user_attrs.get('win_rate', '?')}%")

    print(f"\n  Top 15 trials:")
    sorted_trials = sorted(valid_trials, key=lambda t: t.value, reverse=True)
    for t in sorted_trials[:15]:
        print(f"    #{t.number:2d}: score={t.value:+6.2f} "
              f"bp={t.params['bust_penalty']:5.1f} "
              f"fl_s={t.params['fl_ev_scale']:.2f} "
              f"bust={t.user_attrs.get('bust_rate', -1):5.1f}% "
              f"fl={t.user_attrs.get('fl_rate', -1):5.1f}% "
              f"win={t.user_attrs.get('win_rate', -1):5.1f}%")


if __name__ == "__main__":
    main()
