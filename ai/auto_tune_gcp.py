"""
Optuna GCP parallel tuning: bust_penalty × fl_ev_scale → maximize score.

Each trial:
  1. Optuna suggests (bust_penalty, fl_ev_scale)
  2. SSH to a free GCP VM → run benchmark (100 games)
  3. Parse total_score → return as objective

Usage:
    python ai/auto_tune_gcp.py --trials 50 --games 100

Prerequisites:
    - GCP VMs running with binary + models deployed
    - pip install optuna
"""
import argparse
import subprocess
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import optuna

# GCP VM pool (14 VMs, VM-5 excluded due to capacity)
VM_POOL = [
    ("ofc-selfplay-0", "us-central1-a"),
    ("ofc-selfplay-1", "us-central1-b"),
    ("ofc-selfplay-2", "us-east1-b"),
    ("ofc-selfplay-3", "us-west1-b"),
    ("ofc-selfplay-4", "europe-west1-b"),
    ("ofc-selfplay-6", "us-central1-f"),
    ("ofc-selfplay-7", "us-east1-c"),
    ("ofc-selfplay-8", "us-east1-d"),
    ("ofc-selfplay-9", "us-west1-a"),
    ("ofc-selfplay-10", "us-west1-c"),
    ("ofc-selfplay-11", "europe-west1-c"),
    ("ofc-selfplay-12", "europe-west1-d"),
    ("ofc-selfplay-13", "us-east4-a"),
    ("ofc-selfplay-14", "us-east4-c"),
]

# Thread-safe VM allocation
vm_lock = threading.Lock()
vm_available = list(range(len(VM_POOL)))


def acquire_vm():
    """Get a free VM index, blocking until one is available."""
    while True:
        with vm_lock:
            if vm_available:
                return vm_available.pop(0)
        time.sleep(1)


def release_vm(idx):
    """Return a VM to the pool."""
    with vm_lock:
        vm_available.append(idx)


def run_benchmark_on_vm(vm_idx, bust_penalty, fl_ev_scale, games, seed):
    """SSH to VM, run benchmark, parse and return total_score."""
    vm_name, zone = VM_POOL[vm_idx]

    remote_cmd = (
        f"cd /home/Owner/rust_benchmark && "
        f"./target/release/ofc_benchmark "
        f"--games {games} --seed {seed} "
        f"--mcts-sims 400 --rollouts 250 "
        f"--bust-penalty {bust_penalty:.1f} "
        f"--fl-ev-scale {fl_ev_scale:.2f} "
        f"--mode benchmark"
    )
    cmd = f'gcloud compute ssh {vm_name} --zone={zone} --command="{remote_cmd}"'

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=1800
        )
        output = result.stdout + result.stderr  # gcloud may mix channels

        # Parse total score
        for line in output.splitlines():
            if "Total score:" in line:
                match = re.search(r'([+-]?\d+\.?\d*)/hand', line)
                if match:
                    score = float(match.group(1))
                    # Also parse other metrics for logging
                    metrics = {"score": score}
                    for l in output.splitlines():
                        if "Bust rate:" in l:
                            m = re.search(r'(\d+\.?\d*)%', l)
                            if m:
                                metrics["bust"] = float(m.group(1))
                        elif "FL entry rate:" in l:
                            m = re.search(r'(\d+\.?\d*)%', l)
                            if m:
                                metrics["fl"] = float(m.group(1))
                        elif "Win rate:" in l:
                            m = re.search(r'(\d+\.?\d*)%', l)
                            if m:
                                metrics["win"] = float(m.group(1))
                    return metrics

        print(f"  [{vm_name}] Failed to parse output")
        return None

    except subprocess.TimeoutExpired:
        print(f"  [{vm_name}] Timeout")
        return None
    except Exception as e:
        print(f"  [{vm_name}] Error: {e}")
        return None


def objective(trial, games, base_seed):
    """Optuna objective: maximize total_score."""
    bust_penalty = trial.suggest_float("bust_penalty", 0.0, 20.0)
    fl_ev_scale = trial.suggest_float("fl_ev_scale", 0.3, 3.0)

    vm_idx = acquire_vm()
    vm_name = VM_POOL[vm_idx][0]
    seed = base_seed + trial.number * 1000

    try:
        print(f"  Trial {trial.number}: bp={bust_penalty:.1f} fl={fl_ev_scale:.2f} → {vm_name}")
        metrics = run_benchmark_on_vm(vm_idx, bust_penalty, fl_ev_scale, games, seed)

        if metrics is None:
            return float("-inf")

        score = metrics["score"]
        trial.set_user_attr("bust_rate", metrics.get("bust", -1))
        trial.set_user_attr("fl_rate", metrics.get("fl", -1))
        trial.set_user_attr("win_rate", metrics.get("win", -1))

        print(f"  Trial {trial.number}: score={score:+.2f} "
              f"bust={metrics.get('bust', '?')}% fl={metrics.get('fl', '?')}% "
              f"win={metrics.get('win', '?')}%")
        return score

    finally:
        release_vm(vm_idx)


def main():
    parser = argparse.ArgumentParser(description="Optuna GCP parallel tuning")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--games", type=int, default=100,
                        help="Games per trial (more = less noise)")
    parser.add_argument("--seed", type=int, default=42000)
    parser.add_argument("--n-vms", type=int, default=10,
                        help="Number of VMs to use")
    parser.add_argument("--study-name", default="ofc_tune_v1")
    parser.add_argument("--db", default="sqlite:///ai/auto_tune_results.db")
    args = parser.parse_args()

    # Limit VM pool
    global VM_POOL, vm_available
    VM_POOL = VM_POOL[:args.n_vms]
    vm_available = list(range(len(VM_POOL)))

    print(f"=== Optuna GCP Tuning ===")
    print(f"  Trials: {args.trials}")
    print(f"  Games/trial: {args.games}")
    print(f"  VMs: {args.n_vms}")
    print(f"  Params: bust_penalty [0, 20], fl_ev_scale [0.3, 3.0]")
    print()

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.db,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Run trials in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.n_vms) as executor:
        futures = []
        for i in range(args.trials):
            trial = study.ask()
            future = executor.submit(objective, trial, args.games, args.seed)
            futures.append((trial, future))

        for trial, future in futures:
            try:
                value = future.result(timeout=3600)
                study.tell(trial, value)
            except Exception as e:
                print(f"  Trial {trial.number} failed: {e}")
                study.tell(trial, float("-inf"))

    # Print results
    print(f"\n{'='*60}")
    print(f"  OPTUNA RESULTS ({len(study.trials)} trials)")
    print(f"{'='*60}")

    best = study.best_trial
    print(f"  Best score: {best.value:+.2f}")
    print(f"  bust_penalty: {best.params['bust_penalty']:.1f}")
    print(f"  fl_ev_scale:  {best.params['fl_ev_scale']:.2f}")
    print(f"  bust_rate: {best.user_attrs.get('bust_rate', '?')}%")
    print(f"  fl_rate:   {best.user_attrs.get('fl_rate', '?')}%")
    print(f"  win_rate:  {best.user_attrs.get('win_rate', '?')}%")

    print(f"\n  Top 10 trials:")
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value else float("-inf"), reverse=True)
    for t in sorted_trials[:10]:
        if t.value is not None:
            print(f"    #{t.number}: score={t.value:+.2f} "
                  f"bp={t.params['bust_penalty']:.1f} "
                  f"fl_s={t.params['fl_ev_scale']:.2f} "
                  f"bust={t.user_attrs.get('bust_rate', '?')}% "
                  f"fl={t.user_attrs.get('fl_rate', '?')}%")


if __name__ == "__main__":
    main()
