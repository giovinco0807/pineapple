import sys
import copy
import time
import os
import glob
from pathlib import Path
from typing import Optional
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.models.networks import PolicyNetwork, ValueNetwork
from ai.training.config import TrainingConfig, TRAINING_CONFIG
from ai.training.selfplay_dataset import SelfPlayDataset, collate_self_play
from ai.engine.action_space import MAX_ACTIONS

class CombinedONNXModel(nn.Module):
    def __init__(self, policy_net, value_net):
        super().__init__()
        self.policy_net = policy_net
        self.value_net = value_net

    def forward(self, state):
        # Return logits (before masking/softmax) and value
        logits = self.policy_net.net(state)
        value = self.value_net(state)["value"]
        return logits, value

def export_to_onnx(policy_net, value_net, out_path, device="cpu"):
    """Export combined networks to a single ONNX file for Rust inference."""
    policy_net.eval()
    value_net.eval()
    
    combined = CombinedONNXModel(policy_net, value_net).to(device)
    dummy_state = torch.zeros(1, 490, device=device)
    
    torch.onnx.export(
        combined,
        (dummy_state,),
        out_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['state'],
        output_names=['logits', 'value'],
        dynamic_axes={
            'state': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        }
    )
    print(f"  [ONNX Export] Saved to {out_path}")


def train_from_dataset(
    policy_net: nn.Module,
    value_net: nn.Module,
    dataloader: DataLoader,
    policy_opt: torch.optim.Optimizer,
    value_opt: torch.optim.Optimizer,
    device: str = "cpu",
) -> dict:
    """
    Update networks from self-play DataLoader.
    """
    policy_net.train()
    value_net.train()

    total_p_loss = 0.0
    total_v_loss = 0.0
    total_batches = 0

    for batch in dataloader:
        batch_states = batch["state_vec"].to(device)
        batch_masks = batch["valid_mask"].to(device)
        batch_targets = batch["targets"].to(device)
        batch_rewards = batch["reward"].to(device)

        # === Policy update ===
        pred_probs = policy_net(batch_states, batch_masks)
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
        v_loss = F.mse_loss(pred["value"].squeeze(-1), batch_rewards)

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
    replay_dir: str = "ai/data/selfplay_replays",
    config: TrainingConfig = TRAINING_CONFIG,
    device: str = "auto",
):
    """
    Main self-play training loop.
    1. Load BC checkpoints
    2. Export initial ONNX model
    3. Loop:
       a. Wait for new JSONL self-play data
       b. Load dataset and train
       c. Export new ONNX model
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    bc_path = Path(bc_dir)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    replay_path = Path(replay_dir)
    replay_path.mkdir(parents=True, exist_ok=True)

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

    # Export initial ONNX for Rust generator
    onnx_path = save_path / "current_model.onnx"
    export_to_onnx(policy_net, value_net, str(onnx_path), device=device)

    print(f"\n=== Starting Self-Play Training Loop ===")
    print(f"  Iterations: {config.sp_iterations}")
    print(f"  Monitoring {replay_dir} for .jsonl files")
    print()

    for iteration in range(config.sp_iterations):
        iter_start = time.time()

        print(f"--- Iteration {iteration+1}/{config.sp_iterations} ---")
        
        # Trigger Rust binary
        rust_cmd = [
            "cargo", "run", "--release", "--bin", "self_play", "--",
            "--model-path", str(onnx_path.resolve()),
            "--output-dir", str(replay_path.resolve()),
            "--games", str(config.sp_games_per_iter),
            "--simulations", str(config.sp_mcts_simulations),
            "--threads", str(config.sp_threads)
        ]
        
        print("  Generating self-play data with Rust MCTS...")
        try:
            mcts_dir = Path(__file__).resolve().parent.parent / "rust_solver" / "mcts_gen"
            subprocess.run(rust_cmd, cwd=str(mcts_dir), check=True)
        except subprocess.CalledProcessError as e:
            print(f"  Rust MCTS generator failed: {e}")
            break
        except FileNotFoundError:
            print("  cargo not found. Make sure Rust is installed.")
            break

        jsonl_files = glob.glob(str(replay_path / "*.jsonl"))
        
        if not jsonl_files:
            print("  No .jsonl trajectories found! Waiting...")
            time.sleep(5)
            continue

        print(f"  Found {len(jsonl_files)} trajectory files.")
        
        # Build Dataset and DataLoader
        dataset = SelfPlayDataset(jsonl_files)
        if len(dataset) == 0:
            print("  Dataset is empty! Skipping.")
            continue
            
        dataloader = DataLoader(
            dataset,
            batch_size=config.bc_batch_size,
            shuffle=True,
            collate_fn=collate_self_play,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )

        # Train
        print("  Training...")
        losses = train_from_dataset(
            policy_net, value_net,
            dataloader,
            policy_opt, value_opt,
            device=device,
        )

        iter_time = time.time() - iter_start
        print(f"  P_loss={losses['policy_loss']:.4f} "
              f"V_loss={losses['value_loss']:.4f} "
              f"Samples={len(dataset)} "
              f"Time={iter_time:.1f}s")

        # Save Checkpoints
        if (iteration + 1) % config.checkpoint_interval == 0:
            torch.save(policy_net.state_dict(),
                       save_path / f"sp_policy_iter{iteration+1}.pt")
            torch.save(value_net.state_dict(),
                       save_path / f"sp_value_iter{iteration+1}.pt")
            print(f"  Saved checkpoint iter {iteration+1}")

        # Export new ONNX for next iteration
        export_to_onnx(policy_net, value_net, str(onnx_path), device=device)
        
        # Sync to GCS
        print("  Syncing artifacts to GCS...")
        try:
            is_win = (sys.platform == "win32")
            subprocess.run(["gsutil", "-m", "rsync", "-r", str(save_path), "gs://ofc-solver-485418/ofc_rl_output/models"], check=False, shell=is_win)
        except FileNotFoundError:
            print("  [!] gsutil not found, skipping GCS sync.")
            
        print()

    # Save final
    torch.save(policy_net.state_dict(), save_path / "sp_policy_final.pt")
    torch.save(value_net.state_dict(), save_path / "sp_value_final.pt")
    print(f"\nTraining complete!")
    print(f"Models saved to {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Self-Play Training (Phase C)")
    parser.add_argument("--bc-dir", default="ai/models/checkpoints",
                        help="BC checkpoint directory")
    parser.add_argument("--save", default="ai/models/selfplay",
                        help="Self-play checkpoint directory")
    parser.add_argument("--replays", default="ai/data/selfplay_replays",
                        help="Directory containing JSONL replays")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = TrainingConfig(
        sp_iterations=args.iterations,
    )
    config.sp_threads = args.threads

    run_self_play_training(
        bc_dir=args.bc_dir,
        save_dir=args.save,
        replay_dir=args.replays,
        config=config,
        device=args.device,
    )

