"""
Automated VN Session Self-Play Training Loop.

Iterates:
  1. Self-Play: Session-based VN-greedy generates hands → JSONL
     (includes FL solver, FL Stay priority, opponent-aware FL placement,
      per-card-count FL bonus with auto-calibration)
  2. Preprocess: JSONL → NPZ (with augmentation)
  3. Train: Retrain VN on original + self-play data
  4. Evaluate: Short session self-play to measure performance
  5. Repeat with improved VN

Usage:
    python -m ai.training.selfplay_loop \
        --base-model models/vn_v3_s200/value_best.pt \
        --base-norm models/vn_v3_s200/norm_stats.json \
        --base-data D:/ofc_data/mc_teacher_s200/processed/mc_teacher.npz \
        --base-data-extra D:/ofc_data/mc_teacher_s50_v2/processed_aug/mc_teacher.npz \
        --output-dir D:/ofc_data/selfplay_loop \
        --rounds 20 --sessions-per-round 200 --eval-sessions 30
"""
import sys
import subprocess
import json
import time
import shutil
from pathlib import Path


def run_cmd(cmd, desc="", log_dir=None):
    """Run command, log to files, return (returncode, stdout_text, stderr_text)."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  CMD: {' '.join(cmd[:8])}...")
    print(f"{'='*60}", flush=True)
    t0 = time.time()

    # Write output to log files to avoid pipe buffer deadlock
    stdout_path = Path(log_dir) / "stdout.log" if log_dir else None
    stderr_path = Path(log_dir) / "stderr.log" if log_dir else None

    if stdout_path:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_f = open(stdout_path, "w", encoding="utf-8")
        stderr_f = open(stderr_path, "w", encoding="utf-8")
    else:
        stdout_f = subprocess.PIPE
        stderr_f = subprocess.PIPE

    import os
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd, stdout=stdout_f, stderr=stderr_f, text=True,
        cwd=str(Path(__file__).parent.parent.parent),
        env=env,
    )
    proc.wait()
    elapsed = time.time() - t0

    if stdout_path:
        stdout_f.close()
        stderr_f.close()
        stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    else:
        stdout_text = proc.stdout.read() if proc.stdout else ""
        stderr_text = proc.stderr.read() if proc.stderr else ""

    if proc.returncode != 0:
        print(f"  ERROR (exit={proc.returncode}, {elapsed:.1f}s):")
        print(stderr_text[-2000:] if stderr_text else "no stderr")
    else:
        lines = (stderr_text or stdout_text or "").strip().split("\n")
        for line in lines[-30:]:
            print(f"  {line}")
        print(f"  ({elapsed:.1f}s)")

    return proc.returncode, stdout_text, stderr_text



def main():
    import argparse
    parser = argparse.ArgumentParser(description="VN Session Self-Play Training Loop")
    parser.add_argument("--base-model", required=True, help="Initial VN model path")
    parser.add_argument("--base-norm", required=True, help="Initial norm_stats.json path")
    parser.add_argument("--base-data", required=True, help="Base training data (NPZ)")
    parser.add_argument("--base-data-extra", default=None, help="Extra training data (NPZ)")
    parser.add_argument("--output-dir", default="D:/ofc_data/selfplay_loop_v2", help="Output directory")
    parser.add_argument("--rounds", type=int, default=20, help="Number of self-play rounds")
    parser.add_argument("--sessions-per-round", type=int, default=200,
                        help="Sessions per self-play round")
    parser.add_argument("--eval-sessions", type=int, default=30, help="Sessions for evaluation")
    parser.add_argument("--temperature", type=float, default=0.1, help="Exploration temperature")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs per round")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--starting-chips", type=int, default=200, help="Starting chips per session")
    parser.add_argument("--max-hands", type=int, default=100, help="Max hands per session")
    parser.add_argument("--resume-round", type=int, default=0, help="Resume from round N")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Track progress
    log_path = out_dir / "loop_log.jsonl"
    history = []

    # Current model starts as base
    current_model = args.base_model
    current_norm = args.base_norm

    # If resuming, find latest model
    if args.resume_round > 0:
        prev_model = out_dir / f"round_{args.resume_round - 1}" / "model" / "value_best.pt"
        prev_norm = out_dir / f"round_{args.resume_round - 1}" / "model" / "norm_stats.json"
        if prev_model.exists():
            current_model = str(prev_model)
            current_norm = str(prev_norm)
            print(f"Resuming from round {args.resume_round}, model: {current_model}")

    print(f"\n{'#'*60}")
    print(f"  VN Session Self-Play Training Loop")
    print(f"{'#'*60}")
    print(f"  Base model:    {current_model}")
    print(f"  Base data:     {args.base_data}")
    print(f"  Extra data:    {args.base_data_extra or 'none'}")
    print(f"  Rounds:        {args.rounds}")
    print(f"  Sessions/rnd:  {args.sessions_per_round}")
    print(f"  Eval sessions: {args.eval_sessions}")
    print(f"  Temperature:   {args.temperature}")
    print(f"  Chips:         {args.starting_chips}")
    print(f"  Output:        {out_dir}")
    print()

    for round_num in range(args.resume_round, args.rounds):
        round_start = time.time()
        round_dir = out_dir / f"round_{round_num}"
        round_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'*'*60}")
        print(f"  ROUND {round_num} / {args.rounds - 1}")
        print(f"  Model: {current_model}")
        print(f"{'*'*60}")

        # ──────────────────────────────────────────────────────
        # Step 1: Session Self-Play
        # ──────────────────────────────────────────────────────
        sp_output = round_dir / "selfplay.jsonl"
        seed = 10000 + round_num * 1000

        rc, _, stderr = run_cmd([
            sys.executable, "-m", "ai.training.selfplay_session",
            "--model", current_model,
            "--norm", current_norm,
            "--sessions", str(args.sessions_per_round),
            "--output", str(sp_output),
            "--temperature", str(args.temperature),
            "--starting-chips", str(args.starting_chips),
            "--max-hands", str(args.max_hands),
            "--seed", str(seed),
        ], f"Round {round_num}: Session Self-Play ({args.sessions_per_round} sessions)",
           log_dir=round_dir / "logs_selfplay")

        if rc != 0:
            print(f"  SELF-PLAY FAILED, stopping loop.")
            break

        # Parse session self-play stats from output (stdout + stderr)
        sp_log = (round_dir / "logs_selfplay" / "stdout.log").read_text(encoding="utf-8", errors="replace") if (round_dir / "logs_selfplay" / "stdout.log").exists() else ""
        sp_stats = {}
        for line in sp_log.split("\n"):
            if "avg_hands=" in line:
                # Parse: 200/200 sessions  (0.4/s)  avg_hands=17.6  fl=7.8%  avg_score=2.5  win=44%
                for part in line.split():
                    if part.startswith("avg_score="):
                        sp_stats["avg_score"] = float(part.split("=")[1])
                    elif part.startswith("fl="):
                        sp_stats["fl_pct"] = float(part.split("=")[1].rstrip("%"))
                    elif part.startswith("win="):
                        sp_stats["win_pct"] = float(part.split("=")[1].rstrip("%"))
                    elif part.startswith("avg_hands="):
                        sp_stats["avg_hands"] = float(part.split("=")[1])
            elif "Total hands:" in line:
                sp_stats["total_hands"] = int(line.split(":")[1].strip().replace(",", ""))
            elif "Total records:" in line:
                sp_stats["total_records"] = int(line.split(":")[1].strip().replace(",", ""))

        # ──────────────────────────────────────────────────────
        # Step 2: Preprocess self-play data
        # ──────────────────────────────────────────────────────
        proc_dir = round_dir / "processed"
        rc, _, _ = run_cmd([
            sys.executable, "ai/training/preprocess_mc_teacher.py",
            str(sp_output),
            "--output", str(proc_dir),
            "--augment",
        ], f"Round {round_num}: Preprocess (augment ×4)",
           log_dir=round_dir / "logs_preprocess")

        if rc != 0:
            print(f"  PREPROCESS FAILED, stopping loop.")
            break

        # ──────────────────────────────────────────────────────
        # Step 3: Train VN on combined data
        # ──────────────────────────────────────────────────────
        model_dir = round_dir / "model"

        # Build data args: base + all previous rounds + current round
        data_args = ["--data", args.base_data]
        if args.base_data_extra:
            data_args += ["--data-extra", args.base_data_extra]

        # Add all self-play rounds as extra data
        all_sp_data = []
        for r in range(round_num + 1):
            sp_npz = out_dir / f"round_{r}" / "processed" / "mc_teacher.npz"
            if sp_npz.exists():
                all_sp_data.append(str(sp_npz))

        for sp_path in all_sp_data:
            data_args += ["--data-extra", sp_path]

        rc, _, stderr = run_cmd([
            sys.executable, "ai/train_value.py",
            *data_args,
            "--epochs", str(args.epochs),
            "--v3",
            "--save", str(model_dir),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--patience", str(args.patience),
            "--bust-weight", "1.0",
            "--fl-weight-loss", "1.0",
        ], f"Round {round_num}: Train VN (base + {len(all_sp_data)} self-play rounds)",
           log_dir=round_dir / "logs_train")

        if rc != 0:
            print(f"  TRAINING FAILED, stopping loop.")
            break

        # Parse training stats from log
        train_log = (round_dir / "logs_train" / "stdout.log").read_text(encoding="utf-8", errors="replace") if (round_dir / "logs_train" / "stdout.log").exists() else ""
        train_stats = {}
        for line in train_log.split("\n"):
            if "Best correlation:" in line:
                train_stats["best_corr"] = float(line.split(":")[-1].strip())
            elif "Best val loss:" in line:
                train_stats["best_val_loss"] = float(line.split(":")[-1].strip())

        # Update current model
        new_model = model_dir / "value_best.pt"
        new_norm = model_dir / "norm_stats.json"
        if new_model.exists():
            current_model = str(new_model)
            current_norm = str(new_norm)
        else:
            print(f"  WARNING: Model not saved, keeping previous model.")

        # ──────────────────────────────────────────────────────
        # Step 4: Evaluate with session self-play
        # ──────────────────────────────────────────────────────
        eval_output = round_dir / "eval.jsonl"
        rc, _, stderr = run_cmd([
            sys.executable, "-m", "ai.training.selfplay_session",
            "--model", current_model,
            "--norm", current_norm,
            "--sessions", str(args.eval_sessions),
            "--output", str(eval_output),
            "--temperature", "0.0",  # Greedy for eval
            "--starting-chips", str(args.starting_chips),
            "--max-hands", str(args.max_hands),
        ], f"Round {round_num}: Evaluate ({args.eval_sessions} sessions, greedy)",
           log_dir=round_dir / "logs_eval")

        # Parse eval stats from log
        eval_log = (round_dir / "logs_eval" / "stdout.log").read_text(encoding="utf-8", errors="replace") if (round_dir / "logs_eval" / "stdout.log").exists() else ""
        eval_stats = {}
        for line in eval_log.split("\n"):
            if "avg_hands=" in line and "sessions" in line:
                for part in line.split():
                    if part.startswith("avg_score="):
                        eval_stats["avg_score"] = float(part.split("=")[1])
                    elif part.startswith("fl="):
                        eval_stats["fl_pct"] = float(part.split("=")[1].rstrip("%"))
                    elif part.startswith("win="):
                        eval_stats["win_pct"] = float(part.split("=")[1].rstrip("%"))
                    elif part.startswith("avg_hands="):
                        eval_stats["avg_hands"] = float(part.split("=")[1])
            elif "Total hands:" in line:
                eval_stats["total_hands"] = int(line.split(":")[1].strip().replace(",", ""))

        round_time = time.time() - round_start

        # ──────────────────────────────────────────────────────
        # Logging
        # ──────────────────────────────────────────────────────
        entry = {
            "round": round_num,
            "selfplay": sp_stats,
            "train": train_stats,
            "eval": eval_stats,
            "model": current_model,
            "elapsed_s": round(round_time, 1),
        }
        history.append(entry)

        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Pretty print progress
        print(f"\n{'─'*60}")
        print(f"  Round {round_num} Summary ({round_time:.0f}s)")
        print(f"{'─'*60}")
        print(f"  Self-Play:  score={sp_stats.get('avg_score', '?'):>6}  "
              f"fl={sp_stats.get('fl_pct', '?')}%  "
              f"win={sp_stats.get('win_pct', '?')}%  "
              f"hands={sp_stats.get('total_hands', '?')}")
        print(f"  Training:   corr={train_stats.get('best_corr', '?')}  "
              f"val_loss={train_stats.get('best_val_loss', '?')}")
        print(f"  Eval:       score={eval_stats.get('avg_score', '?'):>6}  "
              f"fl={eval_stats.get('fl_pct', '?')}%  "
              f"win={eval_stats.get('win_pct', '?')}%")
        print(f"  Model:      {current_model}")

    # ──────────────────────────────────────────────────────
    # Final Summary
    # ──────────────────────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"  SESSION SELF-PLAY LOOP COMPLETE")
    print(f"{'#'*60}")
    print(f"\n  {'Round':>5}  {'EvalScore':>10}  {'FL%':>5}  {'Win%':>5}  "
          f"{'Corr':>6}  {'Time':>6}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*6}")
    for e in history:
        ev = e.get("eval", {})
        tr = e.get("train", {})
        print(f"  {e['round']:>5}  {ev.get('avg_score', '?'):>10}  "
              f"{ev.get('fl_pct', '?'):>5}  "
              f"{ev.get('win_pct', '?'):>5}  "
              f"{tr.get('best_corr', '?'):>6}  "
              f"{e['elapsed_s']:>5.0f}s")

    print(f"\n  Final model: {current_model}")
    print(f"  Log: {log_path}")


if __name__ == "__main__":
    main()
