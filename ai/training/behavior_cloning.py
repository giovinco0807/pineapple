"""
OFC Pineapple - Behavior Cloning Training

Phase B: Train PolicyNetwork + ValueNetwork (Layer 1) from human play data.

Usage:
    python -m ai.training.behavior_cloning --data data/processed --epochs 100
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional

from ai.models.networks import PolicyNetwork, ValueNetwork
from ai.training.config import TrainingConfig, TRAINING_CONFIG


class OFCDataset(Dataset):
    """Dataset for OFC training data (preprocessed numpy arrays)."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.states = np.load(self.data_dir / "states.npy")
        self.actions = np.load(self.data_dir / "actions.npy")
        self.valid_masks = np.load(self.data_dir / "valid_masks.npy")
        self.royalties = np.load(self.data_dir / "royalties.npy")
        self.busted = np.load(self.data_dir / "busted.npy")
        self.fl_entry = np.load(self.data_dir / "fl_entry.npy")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            "state": torch.FloatTensor(self.states[idx]),
            "action_idx": torch.LongTensor([self.actions[idx]])[0],
            "valid_mask": torch.BoolTensor(self.valid_masks[idx]),
            "royalty": torch.FloatTensor([self.royalties[idx]])[0],
            "busted": torch.FloatTensor([self.busted[idx]])[0],
            "fl_entry": torch.FloatTensor([self.fl_entry[idx]])[0],
        }


def train_behavior_cloning(
    data_dir: str,
    config: TrainingConfig = TRAINING_CONFIG,
    save_dir: Optional[str] = None,
    device: str = "auto",
):
    """
    Train Policy + Value networks from preprocessed data.

    Args:
        data_dir: Directory with preprocessed numpy files
        config: Training configuration
        save_dir: Where to save checkpoints (default: ai/models/checkpoints)
        device: 'cpu', 'cuda', or 'auto'
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    dataset = OFCDataset(data_dir)
    n_total = len(dataset)
    n_train = int(0.9 * n_total)
    n_val = n_total - n_train
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=config.bc_batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.bc_batch_size)

    print(f"Data: {n_total} samples (train={n_train}, val={n_val})")

    # Models
    policy_net = PolicyNetwork(
        hidden1=config.hidden1,
        hidden2=config.hidden2,
        max_actions=config.max_actions,
        dropout=config.bc_dropout,
    ).to(device)

    value_net = ValueNetwork(
        hidden1=config.hidden1,
        hidden2=config.hidden2,
    ).to(device)

    policy_opt = torch.optim.Adam(policy_net.parameters(), lr=config.bc_lr)
    value_opt = torch.optim.Adam(value_net.parameters(), lr=config.bc_lr)

    # Training loop
    best_val_acc = 0.0

    for epoch in range(config.bc_epochs):
        policy_net.train()
        value_net.train()
        train_policy_loss = 0.0
        train_value_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            states = batch["state"].to(device)
            action_idx = batch["action_idx"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            royalty = batch["royalty"].to(device)
            busted = batch["busted"].to(device)
            fl_entry = batch["fl_entry"].to(device)

            # === Policy ===
            probs = policy_net(states, valid_mask)
            p_loss = F.nll_loss(torch.log(probs + 1e-8), action_idx)

            policy_opt.zero_grad()
            p_loss.backward()
            policy_opt.step()
            train_policy_loss += p_loss.item() * len(states)

            # Accuracy
            predicted = probs.argmax(dim=-1)
            train_correct += (predicted == action_idx).sum().item()
            train_total += len(states)

            # === Value ===
            pred = value_net(states)
            v_loss = (
                F.mse_loss(pred["royalty_ev"].squeeze(), royalty)
                + F.binary_cross_entropy(pred["bust_prob"].squeeze(), busted)
                + F.binary_cross_entropy(pred["fl_prob"].squeeze(), fl_entry)
            )

            value_opt.zero_grad()
            v_loss.backward()
            value_opt.step()
            train_value_loss += v_loss.item() * len(states)

        # Validation
        policy_net.eval()
        value_net.eval()
        val_correct = 0
        val_top3 = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                states = batch["state"].to(device)
                action_idx = batch["action_idx"].to(device)
                valid_mask = batch["valid_mask"].to(device)

                probs = policy_net(states, valid_mask)
                predicted = probs.argmax(dim=-1)
                val_correct += (predicted == action_idx).sum().item()

                top3 = probs.topk(3, dim=-1).indices
                val_top3 += (top3 == action_idx.unsqueeze(1)).any(dim=-1).sum().item()
                val_total += len(states)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        val_top3_acc = val_top3 / max(val_total, 1)

        if epoch % 10 == 0 or epoch == config.bc_epochs - 1:
            print(
                f"Epoch {epoch:3d}: "
                f"P_loss={train_policy_loss/max(train_total,1):.4f} "
                f"V_loss={train_value_loss/max(train_total,1):.4f} "
                f"Train_Acc={train_acc:.2%} "
                f"Val_Top1={val_acc:.2%} "
                f"Val_Top3={val_top3_acc:.2%}"
            )

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_dir:
                save_path = Path(save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                torch.save(policy_net.state_dict(), save_path / "bc_policy_best.pt")
                torch.save(value_net.state_dict(), save_path / "bc_value_best.pt")

    print(f"\nBest Val Top-1 Accuracy: {best_val_acc:.2%}")

    # Save final
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(policy_net.state_dict(), save_path / "bc_policy_final.pt")
        torch.save(value_net.state_dict(), save_path / "bc_value_final.pt")
        print(f"Models saved to {save_path}")

    return policy_net, value_net


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train BC models")
    parser.add_argument("--data", required=True, help="Preprocessed data directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--save", default="ai/models/checkpoints")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = TrainingConfig(
        bc_lr=args.lr,
        bc_epochs=args.epochs,
        bc_batch_size=args.batch_size,
    )
    train_behavior_cloning(args.data, config, args.save, args.device)
