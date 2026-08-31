"""
VN Self-Play Improvement Loop

Automated loop: Collect data → Train VN → Evaluate → Accept/Reject

Each iteration:
  1. Collect 2000 games with current VN (Rollout evaluator)
  2. Merge new data with accumulated dataset
  3. Train new VN (multi-task: value + bust_prob + fl_prob)
  4. Evaluate with new VN (200 games, Hero=Rollout vs Opp=BC)
  5. If score improves, accept new VN; otherwise keep old

Usage:
    python ai/improve_loop.py --iterations 5
    python ai/improve_loop.py --iterations 3 --collect-games 1000 --eval-games 100
"""

import sys
import os
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run_cmd(cmd, desc=""):
    """Run a subprocess command and stream output."""
    print(f"\n{'─' * 60}")
    print(f"  {desc}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'─' * 60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=False)
    elapsed = time.time() - t0
    print(f"  [{desc}] completed in {elapsed/60:.1f} min (exit={result.returncode})")
    return result.returncode


def collect_data(iteration, games, vn_model, rollouts=150, workers=0):
    """Collect self-play data with current VN."""
    output = f"data/improve_iter{iteration}.npz"
    cmd = [
        sys.executable, "ai/collect_value_data.py",
        "--games", str(games),
        "--rollouts", str(rollouts),
        "--top-k", "20",
        "--vn-top-k", "10",
        "--vn-model", vn_model,
        "--model", "ai/models/selfplay_iter17/bc_policy_best.pt",
        "--output", output,
    ]
    if workers > 0:
        cmd.extend(["--workers", str(workers)])
    rc = run_cmd(cmd, f"Iter {iteration}: Collect {games} games")
    return output if rc == 0 else None


def merge_data(npz_files, output_path):
    """Merge multiple NPZ data files into one."""
    import numpy as np

    all_obs, all_scores, all_turns = [], [], []
    all_busted, all_fl = [], []

    for f in npz_files:
        if not Path(f).exists():
            continue
        data = np.load(f)
        all_obs.append(data['obs'])
        all_scores.append(data['score'])
        all_turns.append(data['turn'])
        if 'busted' in data:
            all_busted.append(data['busted'])
            all_fl.append(data['fl_entry'])
        else:
            # Legacy data without labels: infer from scores
            n = len(data['score'])
            all_busted.append(np.array(data['score'] < -3, dtype=np.bool_))
            all_fl.append(np.array(data['score'] > 15, dtype=np.bool_))

    obs = np.concatenate(all_obs)
    scores = np.concatenate(all_scores)
    turns = np.concatenate(all_turns)
    busted = np.concatenate(all_busted)
    fl = np.concatenate(all_fl)

    np.savez_compressed(output_path, obs=obs, score=scores, turn=turns,
                        busted=busted, fl_entry=fl)
    print(f"  Merged {len(npz_files)} files → {output_path} ({len(obs)} samples)")
    return output_path


def train_vn(data_path, save_dir, pretrained=None, epochs=200, patience=30,
             bust_weight=0.3, fl_weight_loss=0.3):
    """Train VN with multi-task learning."""
    cmd = [
        sys.executable, "ai/train_value.py",
        "--data", data_path,
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--batch-size", "256",
        "--lr", "1e-3",
        "--fl-weight", "5.0",
        "--bust-weight", str(bust_weight),
        "--fl-weight-loss", str(fl_weight_loss),
        "--save", save_dir,
        "--device", "cuda" if is_cuda_available() else "cpu",
    ]
    if pretrained and Path(pretrained).exists():
        cmd.extend(["--pretrained", pretrained])
    rc = run_cmd(cmd, f"Train VN → {save_dir}")
    return rc == 0


def evaluate_vn(vn_model, games=200, seed=42, rollouts=250):
    """Evaluate VN via eval_mcts.py with Rollout for T0 (no MCTS).

    Uses eval_mcts.py with --mcts-sims 0 to use Rollout for T0.
    Hero=Rollout(VN prefilter), Opp=BC greedy, FL resolved by Rust solver.
    """
    cmd = [
        sys.executable, "eval_mcts.py",
        "--games", str(games),
        "--seed", str(seed),
        "--mcts-sims", "0",
        "--t1-rollouts", str(rollouts),
        "--vn-model", vn_model,
    ]
    print(f"\n{'─' * 60}")
    print(f"  Evaluate VN: {vn_model}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'─' * 60}")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    elapsed = time.time() - t0

    # Parse output
    score = None
    bust_rate = None
    fl_rate = None
    for line in result.stdout.split('\n'):
        if 'Total score:' in line:
            # Format: "  Total score:   +8.79 +/- 16.92"
            # We want the first number (mean), not the second (std)
            parts = line.split()
            for p in parts:
                if p.startswith('+') or p.startswith('-'):
                    try:
                        score = float(p)
                        break  # take FIRST number only
                    except ValueError:
                        pass
        if 'Bust rate:' in line:
            for p in line.split():
                if '%' in p:
                    try:
                        bust_rate = float(p.replace('%', ''))
                        break
                    except ValueError:
                        pass
        if 'FL entry rate:' in line:
            for p in line.split():
                if '%' in p:
                    try:
                        fl_rate = float(p.replace('%', ''))
                        break
                    except ValueError:
                        pass

    # Print last part of output
    lines = result.stdout.strip().split('\n')
    for line in lines[-15:]:
        print(f"  {line}")
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-300:]}")

    print(f"  Eval done in {elapsed/60:.1f} min: score={score}, bust={bust_rate}%, fl={fl_rate}%")
    return score, bust_rate, fl_rate


def is_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="VN Improvement Loop")
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--collect-games', type=int, default=2000,
                        help='Games to collect per iteration')
    parser.add_argument('--eval-games', type=int, default=200,
                        help='Games for evaluation')
    parser.add_argument('--rollouts', type=int, default=150,
                        help='Rollouts for data collection')
    parser.add_argument('--eval-rollouts', type=int, default=250,
                        help='Rollouts for evaluation')
    parser.add_argument('--workers', type=int, default=0,
                        help='Workers for data collection (0=auto)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs per iteration')
    parser.add_argument('--bust-weight', type=float, default=0.3)
    parser.add_argument('--fl-weight-loss', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--base-vn', default='ai/models/value_v1/value_best.pt',
                        help='Starting VN model')
    parser.add_argument('--base-data', nargs='*',
                        default=['data/value_data_v4_merged.npz'],
                        help='Existing data files to include')
    args = parser.parse_args()

    print("=" * 60)
    print("  VN Self-Play Improvement Loop")
    print("=" * 60)
    print(f"  Iterations: {args.iterations}")
    print(f"  Collect: {args.collect_games} games/iter (r={args.rollouts})")
    print(f"  Eval: {args.eval_games} games (r={args.eval_rollouts})")
    print(f"  Train: {args.epochs} epochs, bust_w={args.bust_weight}, fl_w={args.fl_weight_loss}")
    print(f"  Base VN: {args.base_vn}")
    print(f"  Base data: {args.base_data}")
    print()

    # History tracking
    history = []
    log_path = ROOT / "ai" / "models" / "improve_log.json"
    current_vn = args.base_vn
    best_score = None

    # Baseline evaluation
    print("\n" + "=" * 60)
    print("  Baseline Evaluation")
    print("=" * 60)
    base_score, base_bust, base_fl = evaluate_vn(
        current_vn, games=args.eval_games, seed=args.seed,
        rollouts=args.eval_rollouts)
    best_score = base_score
    history.append({
        'iteration': 0, 'type': 'baseline',
        'vn_model': current_vn,
        'score': base_score, 'bust_rate': base_bust, 'fl_rate': base_fl,
    })
    print(f"\n  Baseline: score={base_score}, bust={base_bust}%, fl={base_fl}%")

    # Main loop
    all_data_files = list(args.base_data)
    t_total = time.time()

    for iteration in range(1, args.iterations + 1):
        print(f"\n{'=' * 60}")
        print(f"  Iteration {iteration}/{args.iterations}")
        print(f"  Current VN: {current_vn}")
        print(f"  Best score: {best_score}")
        print(f"{'=' * 60}")

        # Step 1: Collect data
        data_file = collect_data(
            iteration, args.collect_games, current_vn,
            rollouts=args.rollouts, workers=args.workers)
        if data_file is None:
            print(f"  ERROR: Data collection failed, skipping iteration")
            continue
        all_data_files.append(data_file)

        # Step 2: Merge data
        merged_path = f"data/improve_merged_iter{iteration}.npz"
        merge_data(all_data_files, merged_path)

        # Step 3: Train new VN
        save_dir = f"ai/models/improve_iter{iteration}"
        ok = train_vn(
            merged_path, save_dir, pretrained=current_vn,
            epochs=args.epochs, bust_weight=args.bust_weight,
            fl_weight_loss=args.fl_weight_loss)
        if not ok:
            print(f"  ERROR: Training failed, skipping iteration")
            continue

        new_vn = f"{save_dir}/value_best.pt"

        # Step 4: Evaluate
        new_score, new_bust, new_fl = evaluate_vn(
            new_vn, games=args.eval_games, seed=args.seed,
            rollouts=args.eval_rollouts)

        # Step 5: Accept/Reject
        accepted = False
        if new_score is not None and best_score is not None:
            if new_score > best_score:
                accepted = True
                print(f"\n  ACCEPTED: {new_score:+.2f} > {best_score:+.2f}")
                best_score = new_score
                current_vn = new_vn
            else:
                print(f"\n  REJECTED: {new_score:+.2f} <= {best_score:+.2f}")
        elif new_score is not None:
            accepted = True
            best_score = new_score
            current_vn = new_vn
            print(f"\n  ACCEPTED (first valid): {new_score:+.2f}")

        history.append({
            'iteration': iteration,
            'type': 'accepted' if accepted else 'rejected',
            'vn_model': new_vn,
            'score': new_score, 'bust_rate': new_bust, 'fl_rate': new_fl,
            'data_files': list(all_data_files),
            'n_total_samples': _count_samples(merged_path),
        })

        # Save log
        with open(log_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"  Log saved to: {log_path}")

    # Summary
    total_time = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"  Improvement Loop Complete")
    print(f"{'=' * 60}")
    print(f"  Total time: {total_time/3600:.1f} hours")
    print(f"  Best VN: {current_vn}")
    print(f"  Best score: {best_score}")
    print(f"\n  History:")
    for h in history:
        status = h['type'].upper()
        print(f"    Iter {h['iteration']}: {status} "
              f"score={h['score']} bust={h['bust_rate']}% fl={h['fl_rate']}% "
              f"model={h['vn_model']}")


def _count_samples(npz_path):
    try:
        import numpy as np
        return len(np.load(npz_path)['obs'])
    except Exception:
        return 0


if __name__ == '__main__':
    main()
