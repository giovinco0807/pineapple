"""
OFC Pineapple - PPO Self-Play Training

Train a PolicyNet via MaskablePPO (sb3-contrib) with self-play.

Features:
  - BC pre-trained model initialization
  - Self-play with periodic opponent model updates
  - Tensorboard logging
  - Checkpoint saving

Usage:
    # Quick test (1000 steps)
    python ai/rl/train_ppo.py --steps 1000 --test

    # Production run with parallel envs (recommended)
    python ai/rl/train_ppo.py --steps 1000000 --bc-init ai/models/selfplay_iter17/bc_policy_best.pt --n-envs 8

    # Resume from checkpoint
    python ai/rl/train_ppo.py --resume ai/models/ppo_selfplay/latest.zip --steps 5000000
"""
import sys
import os
import argparse
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from ai.rl.ofc_selfplay_env import OFCSelfPlayEnv
from ai.engine.encoding import STATE_DIM
from ai.engine.action_space import MAX_ACTIONS


def make_env(opponent_mode="random", seed=None, fl_entry_bonus=15.0):
    """Factory function for creating environments."""
    def _init():
        env = OFCSelfPlayEnv(
            opponent_mode=opponent_mode,
            fl_entry_bonus=fl_entry_bonus,
            seed=seed,
        )
        return env
    return _init


def load_bc_weights_into_ppo(ppo_model, bc_checkpoint_path: str):
    """
    Initialize PO policy network from a BC-trained PolicyNetwork checkpoint.

    The BC model has architecture: 520 → 1024 → 512 → 256 → 250
    The PPO MlpPolicy's pi_net needs to match this structure.
    """
    checkpoint = torch.load(bc_checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        bc_state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        bc_state = checkpoint["state_dict"]
    else:
        bc_state = checkpoint

    # Map BC weights → PPO policy network
    # BC model:  net.0 (520→1024), net.3 (1024→512), net.6 (512→256), net.9 (256→250)
    #   (gaps are ReLU=1, Dropout=2, then next Linear)
    # PPO model: mlp_extractor.policy_net.0 (520→1024), .2 (1024→512), .4 (512→256)
    #            action_net (256→250)

    ppo_policy = ppo_model.policy
    ppo_state = ppo_policy.state_dict()

    # Explicit mapping: BC key → PPO key
    explicit_mapping = {
        # Hidden layers (BC net.0/3/6 → PPO policy_net.0/2/4)
        "net.0.weight": "mlp_extractor.policy_net.0.weight",
        "net.0.bias":   "mlp_extractor.policy_net.0.bias",
        "net.3.weight": "mlp_extractor.policy_net.2.weight",
        "net.3.bias":   "mlp_extractor.policy_net.2.bias",
        "net.6.weight": "mlp_extractor.policy_net.4.weight",
        "net.6.bias":   "mlp_extractor.policy_net.4.bias",
        # Output layer (BC net.9 → PPO action_net)
        "net.9.weight": "action_net.weight",
        "net.9.bias":   "action_net.bias",
    }

    loaded = 0
    skipped = 0

    for bc_key, ppo_key in explicit_mapping.items():
        if bc_key in bc_state and ppo_key in ppo_state:
            bc_shape = bc_state[bc_key].shape
            ppo_shape = ppo_state[ppo_key].shape
            if bc_shape == ppo_shape:
                ppo_state[ppo_key] = bc_state[bc_key].clone()
                loaded += 1
                print(f"    {bc_key} → {ppo_key} ({list(bc_shape)})")
            else:
                print(f"    SKIP {bc_key}: shape mismatch {bc_shape} vs {ppo_shape}")
                skipped += 1
        else:
            skipped += 1

    if loaded > 0:
        ppo_policy.load_state_dict(ppo_state, strict=False)
        print(f"  BC → PPO: loaded {loaded}/8 params, skipped {skipped}")
    else:
        print(f"  WARNING: Could not map BC weights to PPO. Training from scratch.")

    return loaded > 0


def create_ppo_model(env, args):
    """Create MaskablePPO model with proper configuration."""
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    except ImportError:
        print("ERROR: sb3-contrib is required. Install with:")
        print("  pip install sb3-contrib stable-baselines3")
        sys.exit(1)

    # Policy architecture matching BC: 520 → 1024 → 512 → 256
    policy_kwargs = dict(
        net_arch=dict(
            pi=[1024, 512, 256],
            vf=[512, 256, 128],
        ),
        activation_fn=nn.ReLU,
    )

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(ROOT / "ai" / "logs" / "ppo_selfplay"),
        verbose=1,
        seed=args.seed,
    )

    return model


def main():
    parser = argparse.ArgumentParser(description="OFC Pineapple PPO Self-Play Training")

    # Core params
    parser.add_argument("--steps", type=int, default=1_000_000,
                        help="Total training timesteps")
    parser.add_argument("--bc-init", type=str, default=None,
                        help="Path to BC checkpoint for initialization")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to PPO checkpoint to resume from")
    parser.add_argument("--save-dir", type=str, default="ai/models/ppo_selfplay",
                        help="Directory to save checkpoints")

    # PPO hyperparams
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Steps per rollout (per env)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Minibatch size")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="PPO epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="Entropy coefficient (exploration)")
    parser.add_argument("--seed", type=int, default=42)

    # Environment
    parser.add_argument("--opponent", type=str, default="random",
                        choices=["random", "self", "model"],
                        help="Opponent type")
    parser.add_argument("--fl-bonus", type=float, default=15.0,
                        help="FL entry bonus reward")
    parser.add_argument("--bust-penalty", type=float, default=10.0,
                        help="Bust penalty")
    parser.add_argument("--fl-progress-scale", type=float, default=1.0,
                        help="FL progress shaping reward scale (0=disabled)")
    parser.add_argument("--fl-force-ratio", type=float, default=0.0,
                        help="Fraction of episodes with forced FL-seeking T0 (0-1)")
    parser.add_argument("--n-envs", type=int, default=1,
                        help="Number of parallel environments (use 4-8 for speed)")
    parser.add_argument("--bc-opponent", type=str, default=None,
                        help="Path to BC model for opponent (enables model opponent mode)")

    # Eval & logging
    parser.add_argument("--eval-freq", type=int, default=10000,
                        help="Evaluate every N steps")
    parser.add_argument("--save-freq", type=int, default=50000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--test", action="store_true",
                        help="Quick test mode (1000 steps)")

    args = parser.parse_args()

    if args.test:
        args.steps = 1000
        args.n_steps = 128
        args.batch_size = 64
        args.eval_freq = 500
        args.save_freq = 500

    # Auto-set opponent mode if bc-opponent is provided
    if args.bc_opponent and args.opponent == "random":
        args.opponent = "model"

    save_dir = ROOT / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  OFC Pineapple - PPO Self-Play Training")
    print("=" * 60)
    print(f"  Steps:        {args.steps:,}")
    print(f"  BC init:      {args.bc_init or 'None (random init)'}")
    print(f"  Opponent:     {args.opponent}")
    print(f"  Parallel:     {args.n_envs} envs")
    print(f"  LR:           {args.lr}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  FL bonus:     {args.fl_bonus}")
    print(f"  Save dir:     {save_dir}")
    print("=" * 60)

    # Create environment(s)
    if args.n_envs > 1:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        from sb3_contrib.common.wrappers import ActionMasker

        def mask_fn(env):
            return env.action_masks()

        def make_masked_env(seed_offset):
            def _init():
                e = OFCSelfPlayEnv(
                    opponent_mode=args.opponent,
                    opponent_model_path=args.bc_opponent,
                    fl_entry_bonus=args.fl_bonus,
                    bust_penalty=args.bust_penalty,
                    fl_progress_scale=args.fl_progress_scale,
                    fl_force_ratio=args.fl_force_ratio,
                    seed=args.seed + seed_offset if args.seed else None,
                )
                e = ActionMasker(e, mask_fn)
                return e
            return _init

        env = SubprocVecEnv(
            [make_masked_env(i) for i in range(args.n_envs)],
            start_method="spawn",
        )
        print(f"  Created {args.n_envs} parallel environments (SubprocVecEnv)")
    else:
        env = OFCSelfPlayEnv(
            opponent_mode=args.opponent,
            opponent_model_path=args.bc_opponent,
            fl_entry_bonus=args.fl_bonus,
            bust_penalty=args.bust_penalty,
            fl_progress_scale=args.fl_progress_scale,
            fl_force_ratio=args.fl_force_ratio,
            seed=args.seed,
        )

    # Create or load model
    if args.resume:
        from sb3_contrib import MaskablePPO
        print(f"\n  Resuming from: {args.resume}")
        model = MaskablePPO.load(
            args.resume,
            env=env,
            tensorboard_log=str(ROOT / "ai" / "logs" / "ppo_selfplay"),
        )
    else:
        model = create_ppo_model(env, args)

        # Initialize from BC
        if args.bc_init:
            print(f"\n  Initializing from BC: {args.bc_init}")
            bc_path = ROOT / args.bc_init if not Path(args.bc_init).is_absolute() else Path(args.bc_init)
            if bc_path.exists():
                load_bc_weights_into_ppo(model, str(bc_path))
            else:
                print(f"  WARNING: BC checkpoint not found: {bc_path}")

    # Callbacks
    from stable_baselines3.common.callbacks import (
        CheckpointCallback, EvalCallback, CallbackList
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(save_dir),
        name_prefix="ppo_ofc",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # Custom callback for logging hand metrics
    class HandMetricsCallback(CheckpointCallback):
        """Log custom OFC metrics to tensorboard."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.busts = 0
            self.fl_entries = 0
            self.total_hands = 0
            self.total_reward = 0.0

        def _on_step(self) -> bool:
            # Check infos for hand results
            infos = self.locals.get("infos", [])
            for info in infos:
                if "hand_result" in info:
                    hr = info["hand_result"]
                    self.total_hands += 1
                    if hr["busted"][0]:
                        self.busts += 1
                    if hr["fl_entry"][0]:
                        self.fl_entries += 1
                    self.total_reward += hr["raw_score"][0]

                    # Log periodically
                    if self.total_hands % 100 == 0 and self.total_hands > 0:
                        bust_rate = self.busts / self.total_hands * 100
                        fl_rate = self.fl_entries / self.total_hands * 100
                        avg_score = self.total_reward / self.total_hands

                        self.logger.record("ofc/bust_rate", bust_rate)
                        self.logger.record("ofc/fl_rate", fl_rate)
                        self.logger.record("ofc/avg_score", avg_score)
                        self.logger.record("ofc/total_hands", self.total_hands)

            return True

    metrics_cb = HandMetricsCallback(
        save_freq=999_999_999,  # don't actually save from this callback
        save_path=str(save_dir),
        name_prefix="metrics",
    )

    callbacks = CallbackList([checkpoint_cb, metrics_cb])

    # Train!
    t_start = time.time()
    print(f"\n  Training for {args.steps:,} steps...")
    model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        progress_bar=True,
    )

    elapsed = time.time() - t_start
    print(f"\n  Training complete! ({elapsed:.0f}s = {elapsed/60:.1f}min)")

    # Save final model
    final_path = save_dir / "final"
    model.save(str(final_path))
    print(f"  Saved final model: {final_path}.zip")

    # Print summary
    print(f"\n  === Training Summary ===")
    print(f"    Total hands:  {metrics_cb.total_hands}")
    if metrics_cb.total_hands > 0:
        print(f"    Bust rate:    {metrics_cb.busts/metrics_cb.total_hands*100:.1f}%")
        print(f"    FL rate:      {metrics_cb.fl_entries/metrics_cb.total_hands*100:.1f}%")
        print(f"    Avg score:    {metrics_cb.total_reward/metrics_cb.total_hands:.2f}")

    env.close()


if __name__ == "__main__":
    main()
