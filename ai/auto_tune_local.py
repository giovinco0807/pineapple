"""
Optuna local tuning: bust_penalty × fl_ev_scale → maximize score.
Single process, sequential trials. No SSH, no GCP.

Usage:
    python -u ai/auto_tune_local.py --trials 30 --games 300
"""
import argparse
import subprocess
import re
import time

import optuna


def run_benchmark(bust_penalty, fl_ev_scale, games, seed, binary):
    """Run ofc_benchmark locally, parse results."""
    cmd = [
        binary,
        "--games", str(games),
        "--seed", str(seed),
        "--mcts-sims", "400",
        "--rollouts", "250",
        "--bust-penalty", f"{bust_penalty:.2f}",
        "--fl-ev-scale", f"{fl_ev_scale:.2f}",
        "--mode", "benchmark",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=7200, encoding='utf-8', errors='replace'
        )
        output = result.stdout + result.stderr

        metrics = {}
        for line in output.splitlines():
            if "Total score:" in line:
                m = re.search(r'([+-]?\d+\.?\d*)/hand', line)
                if m:
                    metrics["score"] = float(m.group(1))
            elif "Bust" in line and "%" in line:
                m = re.search(r'\[(\d+\.?\d*)%', line)
                if not m:
                    m = re.search(r'Bust rate:\s*(\d+\.?\d*)%', line)
                if m:
                    metrics["bust"] = float(m.group(1))
            elif "FL entry" in line:
                m = re.search(r'(\d+\.?\d*)%', line)
                if m:
                    metrics["fl"] = float(m.group(1))
            elif "Win rate:" in line:
                m = re.search(r'(\d+\.?\d*)%', line)
                if m:
                    metrics["win"] = float(m.group(1))

        if "score" in metrics:
            return metrics
        print(f"  Failed to parse output")
        return None

    except subprocess.TimeoutExpired:
        print(f"  Timeout (2h)")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Optuna local tuning")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--seed", type=int, default=500000)
    parser.add_argument("--study-name", default="ofc_tune_local")
    parser.add_argument("--db", default="sqlite:///ai/auto_tune_results.db")
    parser.add_argument("--binary",
                        default="ai/rust_benchmark/target/release/ofc_benchmark.exe")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    print(f"=== Optuna Local Tuning ===")
    print(f"  Trials: {args.trials}")
    print(f"  Games/trial: {args.games}")
    print(f"  Params: bust_penalty [0, 20], fl_ev_scale [0.3, 3.0]")
    print()

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.db,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    if not args.resume:
        study.enqueue_trial({"bust_penalty": 0.0, "fl_ev_scale": 1.0})
        study.enqueue_trial({"bust_penalty": 5.0, "fl_ev_scale": 2.0})
        study.enqueue_trial({"bust_penalty": 10.0, "fl_ev_scale": 1.5})
        study.enqueue_trial({"bust_penalty": 17.0, "fl_ev_scale": 2.0})

    def objective(trial):
        bust_penalty = trial.suggest_float("bust_penalty", 0.0, 20.0)
        fl_ev_scale = trial.suggest_float("fl_ev_scale", 0.3, 3.0)
        seed = args.seed + trial.number * 10000

        t0 = time.time()
        print(f"  Trial {trial.number}: bp={bust_penalty:.1f} fl={fl_ev_scale:.2f} "
              f"({args.games}g)...", end="", flush=True)
        metrics = run_benchmark(bust_penalty, fl_ev_scale, args.games, seed, args.binary)
        elapsed = time.time() - t0

        if metrics is None:
            print(f" FAILED ({elapsed:.0f}s)")
            return float("-inf")

        score = metrics["score"]
        trial.set_user_attr("bust_rate", metrics.get("bust", -1))
        trial.set_user_attr("fl_rate", metrics.get("fl", -1))
        trial.set_user_attr("win_rate", metrics.get("win", -1))

        print(f" score={score:+.2f} bust={metrics.get('bust','?')}% "
              f"fl={metrics.get('fl','?')}% win={metrics.get('win','?')}% "
              f"({elapsed:.0f}s)", flush=True)
        return score

    study.optimize(objective, n_trials=args.trials)

    # Print results
    print(f"\n{'='*60}")
    print(f"  RESULTS ({len(study.trials)} trials, {args.games} games/trial)")
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
