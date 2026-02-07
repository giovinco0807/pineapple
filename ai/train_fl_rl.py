"""
Fantasyland RL Training Script

Uses PPO with action masking to train an agent for FL placement.
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

# Check if sb3-contrib is available
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    HAS_MASKABLE_PPO = True
except ImportError:
    HAS_MASKABLE_PPO = False
    print("Warning: sb3-contrib not installed. Install with: pip install sb3-contrib")

from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from ai.fl_rl_env import FantasylandEnv


def mask_fn(env):
    """Return action mask for the environment."""
    return env.action_masks()


def make_env(seed: int = 0, num_cards: int = 14):
    """Create a wrapped environment."""
    def _init():
        env = FantasylandEnv(num_cards=num_cards)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def train(args):
    """Train the FL RL agent."""
    if not HAS_MASKABLE_PPO:
        print("Error: sb3-contrib required. Install with: pip install sb3-contrib")
        return
    
    print(f"Training FL RL Agent")
    print(f"  Total steps: {args.total_steps:,}")
    print(f"  N envs: {args.n_envs}")
    print(f"  Num cards: {args.num_cards}")
    print()
    
    # Create environments
    if args.n_envs > 1:
        env = SubprocVecEnv([make_env(seed=i, num_cards=args.num_cards) for i in range(args.n_envs)])
    else:
        env = DummyVecEnv([make_env(seed=0, num_cards=args.num_cards)])
    
    # Create eval environment
    eval_env = DummyVecEnv([make_env(seed=999, num_cards=args.num_cards)])
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq // args.n_envs,
        save_path=args.output_dir,
        name_prefix="fl_rl"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.output_dir,
        log_path=args.output_dir,
        eval_freq=args.eval_freq // args.n_envs,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
    )
    
    # Policy kwargs
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.ReLU,
    )
    
    # Create or load model
    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = MaskablePPO.load(args.resume, env=env)
        model.learning_rate = args.lr
    else:
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            verbose=1,
            tensorboard_log=args.output_dir,
        )
    
    print(f"Model policy: {model.policy}")
    print(f"Total parameters: {sum(p.numel() for p in model.policy.parameters()):,}")
    print()
    
    # Train
    model.learn(
        total_timesteps=args.total_steps,
        callback=[checkpoint_callback],
        progress_bar=True,
    )
    
    # Save final model
    final_path = os.path.join(args.output_dir, "fl_rl_final")
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}")
    
    env.close()
    eval_env.close()


def evaluate(args):
    """Evaluate a trained model."""
    if not HAS_MASKABLE_PPO:
        print("Error: sb3-contrib required")
        return
    
    print(f"Evaluating model: {args.model_path}")
    
    # Load model
    model = MaskablePPO.load(args.model_path)
    
    # Create environment
    env = FantasylandEnv(num_cards=args.num_cards, render_mode="human" if args.render else None)
    env = ActionMasker(env, mask_fn)
    
    # Evaluate
    total_rewards = []
    fl_stays = 0
    busts = 0
    
    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        total_rewards.append(episode_reward)
        
        if episode_reward < -50:
            busts += 1
        elif episode_reward > 30:
            fl_stays += 1
        
        if args.render:
            env.render()
            print(f"Episode {ep+1}: Reward = {episode_reward}")
    
    # Print stats
    print(f"\nEvaluation Results ({args.episodes} episodes):")
    print(f"  Mean reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"  Bust rate: {busts / args.episodes * 100:.1f}%")
    print(f"  FL Stay rate (estimated): {fl_stays / args.episodes * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="FL RL Training")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the agent")
    train_parser.add_argument("--total-steps", type=int, default=1_000_000)
    train_parser.add_argument("--n-envs", type=int, default=4)
    train_parser.add_argument("--num-cards", type=int, default=14)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--n-steps", type=int, default=2048)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--n-epochs", type=int, default=10)
    train_parser.add_argument("--gamma", type=float, default=0.99)
    train_parser.add_argument("--gae-lambda", type=float, default=0.95)
    train_parser.add_argument("--clip-range", type=float, default=0.2)
    train_parser.add_argument("--ent-coef", type=float, default=0.01)
    train_parser.add_argument("--checkpoint-freq", type=int, default=50000)
    train_parser.add_argument("--eval-freq", type=int, default=10000)
    train_parser.add_argument("--n-eval-episodes", type=int, default=50)
    train_parser.add_argument("--output-dir", type=str, default="ai/training/fl_rl")
    train_parser.add_argument("--resume", type=str, default=None, help="Path to model to resume training from")
    
    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a model")
    eval_parser.add_argument("--model-path", type=str, required=True)
    eval_parser.add_argument("--episodes", type=int, default=100)
    eval_parser.add_argument("--num-cards", type=int, default=14)
    eval_parser.add_argument("--render", action="store_true")
    
    args = parser.parse_args()
    
    if args.command == "train":
        os.makedirs(args.output_dir, exist_ok=True)
        train(args)
    elif args.command == "eval":
        evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
