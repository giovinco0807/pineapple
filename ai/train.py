"""
Training script for OFC Pineapple RL agent.

Uses stable-baselines3 with PPO and action masking.
"""

import os
import argparse
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.rl_env import OFCPineappleEnv


def make_env(rank: int = 0, seed: int = 0):
    """Create a wrapped environment."""
    def _init():
        env = OFCPineappleEnv()
        env.reset(seed=seed + rank)
        return env
    return _init


def train(
    total_timesteps: int = 1_000_000,
    n_envs: int = 4,
    save_freq: int = 50_000,
    eval_freq: int = 25_000,
    model_dir: str = "ai/models",
    log_dir: str = "ai/logs",
):
    """
    Train the RL agent.
    
    Args:
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        save_freq: Save model every N steps
        eval_freq: Evaluate every N steps
        model_dir: Directory to save models
        log_dir: Directory for tensorboard logs
    """
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_ofc_{timestamp}"
    
    print(f"Starting training run: {run_name}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    
    # Create vectorized environment
    env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(100)])
    
    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=log_dir,
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,
        save_path=model_dir,
        name_prefix=run_name,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{model_dir}/best_{run_name}",
        log_path=log_dir,
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=20,
        deterministic=True,
    )
    
    # Train
    print("\nStarting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
    )
    
    # Save final model
    final_path = f"{model_dir}/{run_name}_final"
    model.save(final_path)
    print(f"\nTraining complete! Final model saved to: {final_path}")
    
    return model


def evaluate(model_path: str, n_episodes: int = 100):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to saved model
        n_episodes: Number of episodes to evaluate
    """
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    env = OFCPineappleEnv()
    
    wins = 0
    losses = 0
    draws = 0
    total_reward = 0
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        total_reward += episode_reward
        
        if episode_reward > 0:
            wins += 1
        elif episode_reward < 0:
            losses += 1
        else:
            draws += 1
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}: Reward = {episode_reward:.1f}")
    
    print(f"\nEvaluation Results ({n_episodes} episodes):")
    print(f"  Wins: {wins} ({wins/n_episodes*100:.1f}%)")
    print(f"  Losses: {losses} ({losses/n_episodes*100:.1f}%)")
    print(f"  Draws: {draws} ({draws/n_episodes*100:.1f}%)")
    print(f"  Average Reward: {total_reward/n_episodes:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Train OFC Pineapple RL Agent")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--model", type=str, default=None, help="Model path for evaluation")
    parser.add_argument("--episodes", type=int, default=100)
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train(
            total_timesteps=args.timesteps,
            n_envs=args.envs,
        )
    elif args.mode == "eval":
        if args.model is None:
            print("Error: --model required for evaluation")
            return
        evaluate(args.model, args.episodes)


if __name__ == "__main__":
    main()
