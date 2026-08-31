"""
OFC Pineapple - Curriculum PPO Training

3-phase curriculum with FL experience injection:
  Phase 1: FL-focused (fl_force=0.4, fl_bonus=100, no bust penalty)
  Phase 2: Balance   (fl_force=0.1, fl_bonus=30,  bust_penalty=10)
  Phase 3: Standalone (fl_force=0.0, fl_bonus=30,  bust_penalty=10)

Each phase resumes from the previous phase's final model.

Usage:
    python ai/rl/train_curriculum.py --bc-init ai/models/selfplay_iter17/bc_policy_best.pt
"""
import sys
import subprocess
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


PHASES = [
    {
        "name": "phase1_fl_focus",
        "steps": 2_000_000,
        "fl_force_ratio": 0.4,
        "fl_bonus": 100,
        "bust_penalty": 0,
        "fl_progress_scale": 2.0,
        "ent_coef": 0.1,
        "lr": 1e-4,
    },
    {
        "name": "phase2_balance",
        "steps": 2_000_000,
        "fl_force_ratio": 0.1,
        "fl_bonus": 30,
        "bust_penalty": 10,
        "fl_progress_scale": 1.0,
        "ent_coef": 0.05,
        "lr": 5e-5,
    },
    {
        "name": "phase3_standalone",
        "steps": 1_000_000,
        "fl_force_ratio": 0.0,
        "fl_bonus": 30,
        "bust_penalty": 10,
        "fl_progress_scale": 1.0,
        "ent_coef": 0.03,
        "lr": 3e-5,
    },
]


def run_phase(phase_cfg, bc_init=None, resume_from=None, n_envs=8,
              save_base="ai/models/ppo_curriculum"):
    """Run one training phase."""
    name = phase_cfg["name"]
    save_dir = f"{save_base}/{name}"

    cmd = [
        sys.executable, "ai/rl/train_ppo.py",
        "--steps", str(phase_cfg["steps"]),
        "--n-envs", str(n_envs),
        "--ent-coef", str(phase_cfg["ent_coef"]),
        "--lr", str(phase_cfg["lr"]),
        "--fl-bonus", str(phase_cfg["fl_bonus"]),
        "--bust-penalty", str(phase_cfg["bust_penalty"]),
        "--fl-progress-scale", str(phase_cfg["fl_progress_scale"]),
        "--fl-force-ratio", str(phase_cfg["fl_force_ratio"]),
        "--save-freq", "100000",
        "--save-dir", save_dir,
    ]

    if resume_from:
        cmd.extend(["--resume", resume_from])
    elif bc_init:
        cmd.extend(["--bc-init", bc_init])

    print(f"\n{'#' * 60}")
    print(f"  PHASE: {name}")
    print(f"  Steps: {phase_cfg['steps']:,}")
    print(f"  FL force: {phase_cfg['fl_force_ratio']:.0%}")
    print(f"  FL bonus: {phase_cfg['fl_bonus']}  Bust penalty: {phase_cfg['bust_penalty']}")
    print(f"  Ent coef: {phase_cfg['ent_coef']}  LR: {phase_cfg['lr']}")
    print(f"  Save: {save_dir}")
    if resume_from:
        print(f"  Resume from: {resume_from}")
    elif bc_init:
        print(f"  BC init: {bc_init}")
    print(f"{'#' * 60}\n")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0
    print(f"\n  Phase {name} completed in {elapsed / 60:.1f} min (exit code: {result.returncode})")

    final_path = str(Path(ROOT) / save_dir / "final.zip")
    return final_path


def main():
    parser = argparse.ArgumentParser(description="OFC Pineapple Curriculum Training")
    parser.add_argument("--bc-init", type=str,
                        default="ai/models/selfplay_iter17/bc_policy_best.pt",
                        help="BC model for Phase 1 initialization")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Number of parallel environments")
    parser.add_argument("--save-base", type=str,
                        default="ai/models/ppo_curriculum",
                        help="Base directory for saving all phases")
    parser.add_argument("--start-phase", type=int, default=1,
                        choices=[1, 2, 3],
                        help="Phase to start from (for resuming)")
    parser.add_argument("--resume-model", type=str, default=None,
                        help="Model to resume from when starting mid-curriculum")
    args = parser.parse_args()

    print("=" * 60)
    print("  OFC Pineapple - Curriculum PPO Training")
    print("=" * 60)
    print(f"  BC init:     {args.bc_init}")
    print(f"  Envs:        {args.n_envs}")
    print(f"  Phases:      {len(PHASES)}")
    print(f"  Start phase: {args.start_phase}")
    print("=" * 60)

    t_start = time.time()
    prev_model = args.resume_model

    for i, phase in enumerate(PHASES):
        phase_num = i + 1
        if phase_num < args.start_phase:
            # Skip earlier phases
            prev_model = str(Path(ROOT) / args.save_base / phase["name"] / "final.zip")
            print(f"  Skipping Phase {phase_num} ({phase['name']})")
            continue

        if phase_num == 1 and prev_model is None:
            # Phase 1: BC init
            prev_model = run_phase(
                phase, bc_init=args.bc_init, n_envs=args.n_envs,
                save_base=args.save_base
            )
        else:
            # Phase 2+: resume from previous
            prev_model = run_phase(
                phase, resume_from=prev_model, n_envs=args.n_envs,
                save_base=args.save_base
            )

    total = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  CURRICULUM COMPLETE")
    print(f"  Total time: {total / 60:.1f} min")
    print(f"  Final model: {prev_model}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
