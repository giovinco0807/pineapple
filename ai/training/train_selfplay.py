"""
OFC Pineapple - Self-Play Training (Phase C)

Loads BC checkpoints, runs MCTS self-play loop, updates networks.

Usage:
    python -m ai.training.train_selfplay --bc-dir ai/models/checkpoints --iterations 50
"""
import sys
import copy
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.models.networks import PolicyNetwork, ValueNetwork
from ai.training.config import TrainingConfig, TRAINING_CONFIG
from ai.mcts.mcts import MCTSConfig
from ai.mcts.self_play import generate_self_play_data, TrajectoryStep
from ai.mcts.evaluate import evaluate_models
from ai.engine.action_space import MAX_ACTIONS


def train_from_trajectories(
    policy_net: nn.Module,
    value_net: nn.Module,
    trajectories: list,
    policy_opt: torch.optim.Optimizer,
    value_opt: torch.optim.Optimizer,
    device: str = "cpu",
    batch_size: int = 256,
) -> dict:
    """
    Update networks from self-play data.

    Policy: KL divergence (predict MCTS distribution)
    Value: MSE loss (predict actual reward)
    """
    # Prepare data
    states = np.array([s.state_vec for s in trajectories], dtype=np.float32)
    rewards = np.array([s.reward for s in trajectories], dtype=np.float32)

    n = len(trajectories)
    indices = np.random.permutation(n)

    policy_net.train()
    value_net.train()

    total_p_loss = 0.0
    total_v_loss = 0.0
    total_batches = 0

    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        batch_steps = [trajectories[i] for i in batch_idx]

        # States
        batch_states = torch.FloatTensor(states[batch_idx]).to(device)

        # Rewards
        batch_rewards = torch.FloatTensor(rewards[batch_idx]).to(device)

        # Valid masks (all true for now)
        batch_masks = torch.ones(len(batch_idx), MAX_ACTIONS, dtype=torch.bool).to(device)
        for i, step in enumerate(batch_steps):
            if step.valid_mask is not None:
                batch_masks[i] = torch.BoolTensor(step.valid_mask)

        # MCTS target distributions
        batch_targets = torch.zeros(len(batch_idx), MAX_ACTIONS).to(device)
        for i, step in enumerate(batch_steps):
            for action_idx, prob in step.action_probs.items():
                if action_idx < MAX_ACTIONS:
                    batch_targets[i, action_idx] = prob

        # === Policy update ===
        pred_probs = policy_net(batch_states, batch_masks)
        # KL(target || pred) = sum(target * log(target/pred))
        p_loss = F.kl_div(
            torch.log(pred_probs + 1e-8),
            batch_targets,
            reduction="batchmean",
        )

        policy_opt.zero_grad()
        p_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
        policy_opt.step()

        # === Value update ===
        pred = value_net(batch_states)
        # Combined value target: reward
        v_loss = F.mse_loss(pred["value"].squeeze(), batch_rewards)

        value_opt.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
        value_opt.step()

        total_p_loss += p_loss.item()
        total_v_loss += v_loss.item()
        total_batches += 1

    return {
        "policy_loss": total_p_loss / max(total_batches, 1),
        "value_loss": total_v_loss / max(total_batches, 1),
    }


def run_self_play_training(
    bc_dir: str = "ai/models/checkpoints",
    save_dir: str = "ai/models/selfplay",
    config: TrainingConfig = TRAINING_CONFIG,
    device: str = "auto",
):
    """
    Main self-play training loop.

    1. Load BC checkpoints
    2. For each iteration: self-play -> train -> eval
    3. Save best checkpoint
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    bc_path = Path(bc_dir)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Load BC-trained networks
    policy_net = PolicyNetwork(
        hidden1=config.hidden1,
        hidden2=config.hidden2,
        max_actions=config.max_actions,
    ).to(device)

    value_net = ValueNetwork(
        hidden1=config.hidden1,
        hidden2=config.hidden2,
    ).to(device)

    bc_policy = bc_path / "bc_policy_best.pt"
    bc_value = bc_path / "bc_value_best.pt"

    if bc_policy.exists():
        policy_net.load_state_dict(torch.load(bc_policy, map_location=device))
        print(f"  Loaded BC policy: {bc_policy}")
    else:
        print(f"  WARNING: No BC policy found at {bc_policy}")

    if bc_value.exists():
        value_net.load_state_dict(torch.load(bc_value, map_location=device))
        print(f"  Loaded BC value: {bc_value}")
    else:
        print(f"  WARNING: No BC value found at {bc_value}")

    policy_opt = torch.optim.Adam(policy_net.parameters(), lr=config.sp_lr)
    value_opt = torch.optim.Adam(value_net.parameters(), lr=config.sp_lr)

    # MCTS config
    mcts_config = MCTSConfig(
        num_simulations=config.sp_mcts_simulations,
        c_puct=config.sp_c_puct,
        temperature=config.sp_temperature,
    )

    # Keep a copy for evaluation
    best_win_rate = 0.5
    prev_policy = copy.deepcopy(policy_net.state_dict())
    prev_value = copy.deepcopy(value_net.state_dict())

    print(f"\n=== Starting Self-Play Training ===")
    print(f"  Iterations: {config.sp_iterations}")
    print(f"  Games/iter: {config.sp_games_per_iter}")
    print(f"  MCTS sims: {config.sp_mcts_simulations}")
    print()

    for iteration in range(config.sp_iterations):
        iter_start = time.time()

        print(f"--- Iteration {iteration+1}/{config.sp_iterations} ---")

        # 1. Generate self-play data
        print("  Generating self-play data...")
        trajectories = generate_self_play_data(
            policy_net=policy_net,
            value_net=value_net,
            num_games=config.sp_games_per_iter,
            mcts_config=mcts_config,
            device=device,
        )

        if not trajectories:
            print("  No trajectories! Skipping.")
            continue

        # 2. Train from trajectories
        print("  Training...")
        losses = train_from_trajectories(
            policy_net, value_net,
            trajectories,
            policy_opt, value_opt,
            device=device,
        )

        iter_time = time.time() - iter_start
        print(f"  P_loss={losses['policy_loss']:.4f} "
              f"V_loss={losses['value_loss']:.4f} "
              f"Steps={len(trajectories)} "
              f"Time={iter_time:.1f}s")

        # 3. Evaluate vs previous version
        if (iteration + 1) % config.eval_interval == 0:
            print("  Evaluating vs previous...")

            # Create previous model
            prev_policy_net = PolicyNetwork(
                hidden1=config.hidden1,
                hidden2=config.hidden2,
                max_actions=config.max_actions,
            ).to(device)
            prev_value_net = ValueNetwork(
                hidden1=config.hidden1,
                hidden2=config.hidden2,
            ).to(device)
            prev_policy_net.load_state_dict(prev_policy)
            prev_value_net.load_state_dict(prev_value)

            win_rate = evaluate_models(
                policy_a=policy_net,
                value_a=value_net,
                policy_b=prev_policy_net,
                value_b=prev_value_net,
                num_games=50,
                device=device,
            )

            print(f"  Win rate vs previous: {win_rate:.1%}")

            if win_rate > best_win_rate:
                best_win_rate = win_rate
                prev_policy = copy.deepcopy(policy_net.state_dict())
                prev_value = copy.deepcopy(value_net.state_dict())
                torch.save(policy_net.state_dict(), save_path / "sp_policy_best.pt")
                torch.save(value_net.state_dict(), save_path / "sp_value_best.pt")
                print(f"  ★ New best! Win rate: {win_rate:.1%}")

        # 4. Checkpoint
        if (iteration + 1) % config.checkpoint_interval == 0:
            torch.save(policy_net.state_dict(),
                       save_path / f"sp_policy_iter{iteration+1}.pt")
            torch.save(value_net.state_dict(),
                       save_path / f"sp_value_iter{iteration+1}.pt")
            print(f"  Saved checkpoint iter {iteration+1}")

        print()

    # Save final
    torch.save(policy_net.state_dict(), save_path / "sp_policy_final.pt")
    torch.save(value_net.state_dict(), save_path / "sp_value_final.pt")
    print(f"\nTraining complete! Best win rate: {best_win_rate:.1%}")
    print(f"Models saved to {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Self-Play Training (Phase C)")
    parser.add_argument("--bc-dir", default="ai/models/checkpoints",
                        help="BC checkpoint directory")
    parser.add_argument("--save", default="ai/models/selfplay",
                        help="Self-play checkpoint directory")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = TrainingConfig(
        sp_iterations=args.iterations,
        sp_games_per_iter=args.games,
        sp_mcts_simulations=args.simulations,
    )
    run_self_play_training(
        bc_dir=args.bc_dir,
        save_dir=args.save,
        config=config,
        device=args.device,
    )
