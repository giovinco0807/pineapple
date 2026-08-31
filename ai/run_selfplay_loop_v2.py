"""
Self-Play Loop v2: Rust self-play → BC/VN training → iterate.

Each iteration:
  1. Rust self-play (5000 games, --mode selfplay)
  2. Preprocess JSONL → NPZ
  3. BC train (soft-label, Expectimax anchor + self-play window)
  4. VN train (Expectimax + all self-play accumulated)
  5. Export ONNX
  6. Update FL_EV from logs
  7. Update bust_penalty from logs
  8. Benchmark vs random (100 games, sanity check)
  9. Log metrics

Data mixing strategy:
  - BC: Expectimax seed (anchor) + last 3 iterations of self-play (window)
  - Ratio: iter 1-3: 80:20, iter 4-6: 60:40, iter 7+: 50:50
  - VN: All data accumulated (distribution diversity helps)

Usage:
    python ai/run_selfplay_loop_v2.py --iterations 10 --games 5000
    python ai/run_selfplay_loop_v2.py --iterations 5 --games 1000 --start-iter 1
"""
import sys
import subprocess
import time
import json
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "ai" / "models"
RUST_DIR = ROOT / "ai" / "rust_benchmark"
CONFIG_DIR = ROOT / "ai" / "config"


def run(cmd, label="", timeout=None, cwd=None):
    """Run a command and return stdout."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  $ {cmd}")
    print(f"{'='*60}\n")
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True,
        cwd=str(cwd or ROOT), timeout=timeout,
    )
    if result.stdout:
        print(result.stdout[-2000:])  # Tail to avoid flooding
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-2000:]}")
        raise RuntimeError(f"Command failed (rc={result.returncode}): {cmd}")
    return result.stdout


def get_mix_ratio(iteration):
    """Get Expectimax:self-play ratio based on iteration number."""
    if iteration <= 3:
        return 0.8  # 80% Expectimax, 20% self-play
    elif iteration <= 6:
        return 0.6  # 60:40
    else:
        return 0.5  # 50:50


def main():
    parser = argparse.ArgumentParser(description="Self-Play Loop v2 (Rust)")
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--games', type=int, default=5000,
                        help='Games per self-play iteration')
    parser.add_argument('--mcts-sims', type=int, default=400)
    parser.add_argument('--rollouts', type=int, default=250)
    parser.add_argument('--vn-top-k', type=int, default=10)
    parser.add_argument('--bc-epochs', type=int, default=100)
    parser.add_argument('--vn-epochs', type=int, default=50)
    parser.add_argument('--benchmark-games', type=int, default=100,
                        help='Games for sanity check benchmark')
    parser.add_argument('--start-iter', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)

    # Model paths
    parser.add_argument('--base-bc', default='ai/models/expectimax_bc_v3/bc_policy_best.pt',
                        help='Initial BC model')
    parser.add_argument('--base-vn', default='ai/models/expectimax_vn_v3/value_best.pt',
                        help='Initial VN model')
    parser.add_argument('--expectimax-data', default='data/expectimax_train_v3',
                        help='Expectimax seed data directory (BC npy files)')
    parser.add_argument('--expectimax-vn', default='data/value_data_v4_merged.npz',
                        help='Expectimax VN data (.npz)')

    # BC per-turn models
    parser.add_argument('--bc-t3', default='ai/models/bc_t3/bc_policy_best.pt')
    parser.add_argument('--bc-t4', default='ai/models/bc_t4/bc_policy_best.pt')

    parser.add_argument('--window-size', type=int, default=3,
                        help='Number of recent iterations for BC window')
    args = parser.parse_args()

    # State tracking
    current_bc = args.base_bc
    current_vn = args.base_vn
    all_sp_data = []  # List of (iteration, jsonl_path, npz_dir)
    results_log = []
    bust_penalty = 0.0

    t_start = time.time()
    end_iter = args.start_iter + args.iterations - 1

    for iteration in range(args.start_iter, end_iter + 1):
        print(f"\n{'#'*60}")
        print(f"  ITERATION {iteration}/{end_iter}")
        print(f"  BC: {current_bc}")
        print(f"  VN: {current_vn}")
        print(f"  bust_penalty: {bust_penalty:.1f}")
        print(f"{'#'*60}")

        iter_seed = args.seed + iteration * 10000
        sp_jsonl = DATA_DIR / f"selfplay_v2_iter{iteration}.jsonl"
        sp_npz = DATA_DIR / f"processed_sp_iter{iteration}"
        iter_model_dir = MODELS_DIR / f"selfplay_v2_iter{iteration}"

        # ─── 1. Export ONNX from current models ──────────────────
        run(
            f"python -m ai.training.export_onnx "
            f"--bc {current_bc} --vn {current_vn} "
            f"--bc-t3 {args.bc_t3} --bc-t4 {args.bc_t4} "
            f"--out-dir ai/rust_benchmark/models",
            label=f"[{iteration}] Export ONNX",
        )

        # ─── 2. Rust Self-Play ───────────────────────────────────
        rust_cmd = (
            f"cargo run --release -- "
            f"--games {args.games} --seed {iter_seed} "
            f"--mcts-sims {args.mcts_sims} --rollouts {args.rollouts} "
            f"--vn-top-k {args.vn_top_k} "
            f"--bust-penalty {bust_penalty:.1f} "
            f"--mode selfplay --output {sp_jsonl}"
        )
        run(rust_cmd, label=f"[{iteration}] Self-Play: {args.games} games",
            cwd=str(RUST_DIR), timeout=36000)

        # ─── 3. Preprocess Self-Play Data ─────────────────────────
        run(
            f"python ai/training/preprocess_selfplay.py {sp_jsonl} --output {sp_npz}",
            label=f"[{iteration}] Preprocess self-play data",
        )
        all_sp_data.append((iteration, str(sp_jsonl), str(sp_npz)))

        # ─── 4. BC Training (soft-label, mixed data) ─────────────
        mix_ratio = get_mix_ratio(iteration)
        window_iters = all_sp_data[-args.window_size:]

        # Merge Expectimax + recent self-play data for BC
        bc_merged_dir = DATA_DIR / f"bc_merged_iter{iteration}"
        merge_dirs = []
        if Path(args.expectimax_data).exists():
            merge_dirs.append(args.expectimax_data)
        for _, _, sp_dir in window_iters:
            merge_dirs.append(sp_dir)

        if len(merge_dirs) > 1:
            dirs_list = " ".join(str(d) for d in merge_dirs)
            run(
                f"python -m ai.training.merge_bc_data --dirs {dirs_list} --save {bc_merged_dir}",
                label=f"[{iteration}] Merge BC data ({len(merge_dirs)} sources)",
            )
            bc_data = str(bc_merged_dir)
        else:
            bc_data = merge_dirs[0] if merge_dirs else str(sp_npz)

        bc_cmd = (
            f"python -m ai.training.behavior_cloning "
            f"--data {bc_data} "
            f"--epochs {args.bc_epochs} "
            f"--batch-size 2048 "
            f"--soft-label --soft-temperature 2.0 "
            f"--save {iter_model_dir} "
            f"--skip-vn"
        )
        if current_bc != args.base_bc:
            bc_cmd += f" --pretrained {current_bc}"
        run(bc_cmd, label=f"[{iteration}] BC Training (mix={mix_ratio:.0%} expectimax)")

        # ─── 5. VN: Fixed (no retraining) ─────────────────────────
        # VN stays at base (Expectimax v3, corr=0.921).
        # Selfplay game-outcome labels are too noisy for VN improvement.
        print(f"  [VN] Fixed at {current_vn} (no retraining)")

        # ─── 6. Update FL_EV ─────────────────────────────────────
        all_jsonl_paths = " ".join(str(p) for _, p, _ in all_sp_data)
        try:
            run(
                f"python ai/training/update_fl_ev.py {all_jsonl_paths}",
                label=f"[{iteration}] Update FL_EV",
            )
        except RuntimeError:
            print("  [WARN] FL_EV update failed, keeping defaults")

        # ─── 7. Update bust_penalty ──────────────────────────────
        try:
            bp_out = run(
                f"python ai/training/update_bust_penalty.py {all_jsonl_paths}",
                label=f"[{iteration}] Update bust_penalty",
            )
            # Parse bust_penalty from output
            for line in bp_out.splitlines():
                if "--bust-penalty" in line:
                    try:
                        bust_penalty = float(line.split("--bust-penalty")[1].strip())
                    except (ValueError, IndexError):
                        pass
        except RuntimeError:
            print("  [WARN] bust_penalty update failed, keeping current")

        # ─── 8. Benchmark (sanity check) ─────────────────────────
        new_bc = iter_model_dir / "bc_policy_best.pt"
        new_vn = iter_model_dir / "value_best.pt"  # VN from independent training
        if not new_vn.exists():
            new_vn = Path(current_vn)  # Fallback

        # Export new models to ONNX for benchmark
        run(
            f"python -m ai.training.export_onnx "
            f"--bc {new_bc} --vn {new_vn} "
            f"--bc-t3 {args.bc_t3} --bc-t4 {args.bc_t4} "
            f"--out-dir ai/rust_benchmark/models",
            label=f"[{iteration}] Export ONNX for benchmark",
        )

        bench_cmd = (
            f"cargo run --release -- "
            f"--games {args.benchmark_games} --seed 99999 "
            f"--mcts-sims {args.mcts_sims} --rollouts {args.rollouts} "
            f"--bust-penalty {bust_penalty:.1f} "
            f"--mode benchmark"
        )
        bench_out = run(bench_cmd, label=f"[{iteration}] Benchmark: {args.benchmark_games} games",
                        cwd=str(RUST_DIR), timeout=7200)

        # Parse benchmark metrics
        metrics = parse_benchmark_output(bench_out)

        # ─── 9. Log and accept ────────────────────────────────────
        result = {
            'iteration': iteration,
            'games': args.games,
            'bust_penalty': round(bust_penalty, 1),
            'mix_ratio': mix_ratio,
            'benchmark': metrics,
            'bc_model': str(new_bc),
            'vn_model': str(new_vn),
        }
        results_log.append(result)

        # Always accept (data quality driven, like v1)
        current_bc = str(new_bc)
        if new_vn.exists():
            current_vn = str(new_vn)

        print(f"\n  [OK] Iter {iteration}: "
              f"score={metrics.get('total_score', '?'):+.2f} "
              f"bust={metrics.get('bust_rate', '?'):.1f}% "
              f"fl={metrics.get('fl_rate', '?'):.1f}%")

        # Save progress log
        log_path = MODELS_DIR / "selfplay_v2_log.json"
        with open(log_path, 'w') as f:
            json.dump(results_log, f, indent=2)

        elapsed = time.time() - t_start
        print(f"  Time so far: {elapsed/60:.0f} min")

    # ─── Summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SELF-PLAY LOOP v2 COMPLETE")
    print(f"{'='*60}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Best BC: {current_bc}")
    print(f"  Best VN: {current_vn}")
    for r in results_log:
        m = r['benchmark']
        print(f"  Iter {r['iteration']}: score={m.get('total_score', '?'):+.2f} "
              f"bust={m.get('bust_rate', '?'):.1f}% "
              f"fl={m.get('fl_rate', '?'):.1f}% "
              f"bp={r['bust_penalty']:.1f}")
    print(f"  Total time: {(time.time()-t_start)/60:.0f} min")


def parse_benchmark_output(output):
    """Parse benchmark output for key metrics."""
    metrics = {}
    for line in output.splitlines():
        line = line.strip()
        if "Bust rate:" in line:
            try:
                metrics['bust_rate'] = float(line.split(":")[1].strip().rstrip('%'))
            except (ValueError, IndexError):
                pass
        elif "FL entry rate:" in line:
            try:
                metrics['fl_rate'] = float(line.split(":")[1].strip().rstrip('%'))
            except (ValueError, IndexError):
                pass
        elif "Total score:" in line:
            try:
                parts = line.split(":")[1].strip().split("+/-")
                metrics['total_score'] = float(parts[0].strip())
            except (ValueError, IndexError):
                pass
        elif "Win rate:" in line:
            try:
                metrics['win_rate'] = float(line.split(":")[1].strip().rstrip('%'))
            except (ValueError, IndexError):
                pass
        elif "Speed:" in line:
            try:
                metrics['speed'] = float(line.split(":")[1].strip().rstrip('s/hand'))
            except (ValueError, IndexError):
                pass
    return metrics


if __name__ == '__main__':
    main()
