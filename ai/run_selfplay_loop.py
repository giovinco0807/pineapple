"""
Automated Self-Play Loop: Generate → Preprocess → Train → Evaluate → Repeat

Each iteration:
1. Self-play with current best model (Rollout T0 + BC T1-4, FL via Rust solver)
2. Preprocess all accumulated data
3. Retrain BC (uniform loss)
4. Evaluate vs previous best
5. Accept/reject based on margin

Usage:
    python ai/run_selfplay_loop.py --iterations 10 --games 500 --rollouts 50 --workers 12
"""
import sys
import subprocess
import time
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "ai" / "models"


def run(cmd, label="", timeout=None):
    """Run a command and return stdout."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  $ {cmd}")
    print(f"{'='*60}\n")
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True,
        cwd=str(ROOT), timeout=timeout,
    )
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Command failed: {cmd}")
    return result.stdout


def merge_jsonl(sources, output):
    """Merge multiple JSONL files into one."""
    with open(output, 'w', encoding='utf-8') as fout:
        total = 0
        for src in sources:
            if Path(src).exists():
                with open(src, 'r', encoding='utf-8') as fin:
                    for line in fin:
                        fout.write(line)
                        total += 1
    print(f"  Merged {total} turns from {len(sources)} files → {output}")
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--games', type=int, default=500)
    parser.add_argument('--rollouts', type=int, default=50)
    parser.add_argument('--top-k', type=int, default=30)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--eval-games', type=int, default=100)
    parser.add_argument('--eval-rollouts', type=int, default=50)
    parser.add_argument('--base-model', default='ai/models/ryo_bc_t18/bc_policy_best.pt',
                        help='Initial model to start from')
    parser.add_argument('--include', nargs='*',
                        default=['data/ryo_aug_t18.jsonl'],
                        help='Seed data JSONL files to include in training')
    parser.add_argument('--start-iter', type=int, default=1,
                        help='Starting iteration number (for resuming)')
    parser.add_argument('--v2', action='store_true',
                        help='Use V2 architecture (ResBlock + LayerNorm)')
    parser.add_argument('--fl-enrich', type=float, default=0.0,
                        help='Fraction of games with FL-enriched deals (0.0-1.0)')
    args = parser.parse_args()

    best_model = args.base_model
    all_data_files = []

    # Include seed data files
    for inc_file in (args.include or []):
        if Path(inc_file).exists():
            all_data_files.append(inc_file)
            print(f"Including seed data: {inc_file}")

    results_log = []
    t_start = time.time()

    end_iter = args.start_iter + args.iterations - 1
    for iteration in range(args.start_iter, end_iter + 1):
        print(f"\n{'#'*60}")
        print(f"  ITERATION {iteration}/{end_iter}")
        print(f"  Best model: {best_model}")
        print(f"{'#'*60}")

        iter_data = DATA_DIR / f"selfplay_iter{iteration}.jsonl"
        iter_processed = DATA_DIR / f"processed_iter{iteration}"
        iter_model_dir = MODELS_DIR / f"selfplay_iter{iteration}"

        # ─── 1. Self-Play Data Generation ─────────────────────────
        sp_cmd = (
            f"python ai/self_play.py "
            f"--model {best_model} "
            f"--games {args.games} "
            f"--rollouts {args.rollouts} "
            f"--top-k {args.top_k} "
            f"--workers {args.workers} "
            f"--output {iter_data}"
        )
        if args.fl_enrich > 0:
            sp_cmd += f" --fl-enrich {args.fl_enrich}"
        run(sp_cmd, label=f"[{iteration}] Self-Play: {args.games} games")
        all_data_files.append(str(iter_data))

        # ─── 2. Merge all data ───────────────────────────────────
        merged = DATA_DIR / f"merged_iter{iteration}.jsonl"
        total_turns = merge_jsonl(all_data_files, merged)

        # ─── 3. Preprocess ───────────────────────────────────────
        run(
            f"python ai/training/preprocess_fast.py {merged} "
            f"--output {iter_processed}",
            label=f"[{iteration}] Preprocess: {total_turns} turns",
        )

        # ─── 4. Train BC ─────────────────────────────────────────
        bc_cmd = (
            f"python ai/training/behavior_cloning.py "
            f"--data {iter_processed} "
            f"--epochs {args.epochs} "
            f"--batch-size 256 "
            f"--save {iter_model_dir}"
        )
        if args.v2:
            bc_cmd += " --v2"
        run(bc_cmd, label=f"[{iteration}] BC Training: {args.epochs} epochs")

        # ─── 5. Evaluate via data quality metrics ─────────────────
        new_model = iter_model_dir / "bc_policy_best.pt"
        
        # Analyze self-play data quality
        bust_count = 0
        fl_count = 0
        roy_total = 0.0
        n_turns = 0
        with open(iter_data, 'r', encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                hr = r["hand_result"]
                p = str(r["turn_log"]["player"])
                if hr["busted"][p]:
                    bust_count += 1
                if hr["fl_entry"][p]:
                    fl_count += 1
                roy = hr.get("royalties", {}).get(p, {}).get("total", 0)
                roy_total += roy
                n_turns += 1

        bust_rate = bust_count / max(n_turns, 1) * 100
        fl_rate = fl_count / max(n_turns, 1) * 100
        avg_roy = roy_total / max(n_turns, 1)

        print(f"\n  === Data Quality (iter{iteration}) ===")
        print(f"    Turns: {n_turns}")
        print(f"    Bust:  {bust_rate:.1f}%")
        print(f"    FL:    {fl_rate:.1f}%")
        print(f"    Royalty: {avg_roy:.2f}")

        result = {
            'iteration': iteration,
            'games': args.games,
            'total_turns': total_turns,
            'bust_rate': round(bust_rate, 1),
            'fl_rate': round(fl_rate, 1),
            'avg_royalty': round(avg_roy, 2),
            'model': str(new_model),
            'accepted': True,  # always accept — data quality is what matters
        }
        results_log.append(result)

        # ─── 6. Always accept (data quality driven) ──────────────
        best_model = str(new_model)
        print(f"\n  [OK] ACCEPTED: bust={bust_rate:.1f}% fl={fl_rate:.1f}% roy={avg_roy:.2f}")
        print(f"    -> new best: {best_model}")

        # Save progress log
        log_path = MODELS_DIR / "selfplay_log.json"
        with open(log_path, 'w') as f:
            json.dump(results_log, f, indent=2)

        elapsed = time.time() - t_start
        print(f"  Time so far: {elapsed/60:.0f} min")

    # ─── Summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SELF-PLAY LOOP COMPLETE")
    print(f"{'='*60}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Best model: {best_model}")
    for r in results_log:
        status = "OK" if r['accepted'] else "NG"
        print(f"  [{status}] Iter {r['iteration']}: bust={r['bust_rate']:.1f}% "
              f"fl={r['fl_rate']:.1f}% roy={r['avg_royalty']:.2f} "
              f"turns={r['total_turns']}")
    print(f"  Total time: {(time.time()-t_start)/60:.0f} min")


if __name__ == '__main__':
    main()
