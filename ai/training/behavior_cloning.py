"""
OFC Pineapple - Behavior Cloning Training

Phase B: Train PolicyNetwork + ValueNetwork (Layer 1) from human play data.

Usage:
    python -m ai.training.behavior_cloning --data data/processed --epochs 100
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional

from ai.models.networks import PolicyNetwork, ValueNetwork, PolicyNetworkV2, ValueNetworkV2
from ai.training.config import TrainingConfig, TRAINING_CONFIG


class OFCDataset(Dataset):
    """Dataset for OFC training data (preprocessed numpy arrays).

    Supports optional GPU preloading for faster training.
    """

    def __init__(self, data_dir: str, device: str = "cpu"):
        self.data_dir = Path(data_dir)
        self.device = torch.device(device)

        # Load and convert to tensors (on target device for speed)
        self.states = torch.from_numpy(np.load(self.data_dir / "states.npy").astype(np.float32)).to(self.device)
        self.input_dim = self.states.shape[1]
        self.actions = torch.from_numpy(np.load(self.data_dir / "actions.npy")).long().to(self.device)
        self.valid_masks = torch.from_numpy(np.load(self.data_dir / "valid_masks.npy")).bool().to(self.device)
        # royalties.npy is optional (MC teacher data uses rewards.npy instead)
        royalties_path = self.data_dir / "royalties.npy"
        if royalties_path.exists():
            self.royalties = torch.from_numpy(np.load(royalties_path).astype(np.float32)).to(self.device)
        else:
            self.royalties = None
        self.busted = torch.from_numpy(np.load(self.data_dir / "busted.npy").astype(np.float32)).to(self.device)
        self.fl_entry = torch.from_numpy(np.load(self.data_dir / "fl_entry.npy").astype(np.float32)).to(self.device)
        # Load rewards if available (self-play / MC teacher data)
        rewards_path = self.data_dir / "rewards.npy"
        if rewards_path.exists():
            self.rewards = torch.from_numpy(np.load(rewards_path).astype(np.float32)).to(self.device)
        else:
            self.rewards = torch.zeros(len(self.states), device=self.device)
        # Load action EVs if available (Expectimax data, for soft-label training)
        action_evs_path = self.data_dir / "action_evs.npy"
        if action_evs_path.exists():
            self.action_evs = torch.from_numpy(np.load(action_evs_path).astype(np.float32)).to(self.device)
        else:
            self.action_evs = None

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        item = {
            "state": self.states[idx],
            "action_idx": self.actions[idx],
            "valid_mask": self.valid_masks[idx],
            "royalty": self.royalties[idx],
            "busted": self.busted[idx],
            "fl_entry": self.fl_entry[idx],
            "reward": self.rewards[idx],
        }
        if self.action_evs is not None:
            item["action_evs"] = self.action_evs[idx]
        return item


def train_behavior_cloning(
    data_dir: str,
    config: TrainingConfig = TRAINING_CONFIG,
    save_dir: Optional[str] = None,
    device: str = "auto",
    weighted: bool = False,
    use_v2: bool = False,
    soft_label: bool = False,
    soft_temperature: float = 2.0,
    pretrained: str = None,
    pretrained_vn: str = None,
    skip_vn: bool = False,
    skip_policy: bool = False,
    max_seconds: float = 0.0,
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

    # Load data (all tensors on GPU for speed)
    ds = OFCDataset(data_dir, device=device)
    n_total = len(ds)
    n_train = int(0.9 * n_total)
    n_val = n_total - n_train
    perm = torch.randperm(n_total, device="cpu")
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    bs = config.bc_batch_size

    input_dim = ds.input_dim
    print(f"Data: {n_total} samples (train={n_train}, val={n_val}), input_dim={input_dim}")

    # Models
    if use_v2:
        policy_net = PolicyNetworkV2(
            input_dim=input_dim,
            max_actions=config.max_actions,
            dropout=config.bc_dropout,
        ).to(device)
        value_net = ValueNetworkV2(input_dim=input_dim).to(device)
        print(f"Using V2 architecture (ResBlock)")
    else:
        policy_net = PolicyNetwork(
            input_dim=input_dim,
            max_actions=config.max_actions,
            dropout=config.bc_dropout,
        ).to(device)
        value_net = ValueNetwork(input_dim=input_dim).to(device)

    # Load pretrained weights if specified
    if pretrained and Path(pretrained).exists():
        ck = torch.load(pretrained, map_location=device, weights_only=True)
        sd = ck.get('model_state_dict', ck)
        policy_net.load_state_dict(sd)
        print(f"Loaded pretrained policy from {pretrained}")

    if pretrained_vn and Path(pretrained_vn).exists():
        vk = torch.load(pretrained_vn, map_location=device, weights_only=True)
        vn_sd = vk.get('model_state_dict', vk)
        own_sd = value_net.state_dict()
        loaded = 0
        expanded = 0
        for k, v in vn_sd.items():
            if k not in own_sd:
                continue
            if own_sd[k].shape == v.shape:
                own_sd[k] = v
                loaded += 1
            elif k.endswith('.weight') and len(v.shape) == 2 and \
                 own_sd[k].shape[0] == v.shape[0] and own_sd[k].shape[1] > v.shape[1]:
                # Expand input dim: copy existing cols, zero-init new cols
                own_sd[k][:, :v.shape[1]] = v
                own_sd[k][:, v.shape[1]:] = 0.0
                loaded += 1
                expanded += 1
        value_net.load_state_dict(own_sd)
        msg = f"Loaded pretrained VN from {pretrained_vn} ({loaded}/{len(vn_sd)} params)"
        if expanded:
            msg += f" ({expanded} expanded)"
        print(msg)

    policy_opt = torch.optim.Adam(policy_net.parameters(), lr=config.bc_lr)
    value_opt = torch.optim.Adam(value_net.parameters(), lr=config.bc_lr)

    # Training loop
    best_val_acc = 0.0
    train_start = time.time()

    for epoch in range(config.bc_epochs):
        policy_net.train()
        value_net.train()
        train_policy_loss = 0.0
        train_value_loss = 0.0
        train_correct = 0
        train_total = 0

        # Shuffle train indices each epoch
        shuffled = train_idx[torch.randperm(len(train_idx))]

        for i in range(0, len(shuffled), bs):
            idx = shuffled[i:i+bs]
            states = ds.states[idx]
            action_idx = ds.actions[idx]
            valid_mask = ds.valid_masks[idx]
            royalty = ds.royalties[idx] if ds.royalties is not None else None
            busted = ds.busted[idx]
            fl_entry = ds.fl_entry[idx]

            # === Policy ===
            if not skip_policy:
                probs = policy_net(states, valid_mask)

                if soft_label and ds.action_evs is not None:
                    aev = ds.action_evs[idx]
                    aev = aev.masked_fill(~valid_mask, -1e9)
                    soft_targets = F.softmax(aev / soft_temperature, dim=-1)
                    log_probs = torch.log(probs + 1e-8)
                    p_loss = -(soft_targets * log_probs).sum(dim=-1).mean()
                elif weighted:
                    reward = ds.rewards[idx]
                    w = torch.clamp(reward, min=0.0)
                    w = w / (w.mean() + 1e-8)
                    per_sample_loss = F.nll_loss(
                        torch.log(probs + 1e-8), action_idx, reduction='none'
                    )
                    p_loss = (per_sample_loss * w).mean()
                else:
                    p_loss = F.nll_loss(torch.log(probs + 1e-8), action_idx)

                policy_opt.zero_grad()
                p_loss.backward()
                policy_opt.step()
                train_policy_loss += p_loss.item() * len(states)

            # Accuracy
            if not skip_policy:
                predicted = probs.argmax(dim=-1)
                train_correct += (predicted == action_idx).sum().item()
            train_total += len(states)

            # === Value ===
            if not skip_vn:
                pred = value_net(states)
                v_loss = (
                    F.binary_cross_entropy(pred["bust_prob"].squeeze(-1), busted)
                    + F.binary_cross_entropy(pred["fl_prob"].squeeze(-1), fl_entry)
                )
                if ds.royalties is not None:
                    v_loss = v_loss + F.mse_loss(pred["royalty_ev"].squeeze(-1), royalty)
                # Train value head on rewards (MC score)
                reward = ds.rewards[idx]
                v_loss = v_loss + F.mse_loss(pred["value"].squeeze(-1), reward)

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
            for i in range(0, len(val_idx), bs):
                idx = val_idx[i:i+bs]
                states = ds.states[idx]
                action_idx = ds.actions[idx]
                valid_mask = ds.valid_masks[idx]

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

        if max_seconds and (time.time() - train_start) >= max_seconds:
            print(f"\nReached max_seconds={max_seconds:.0f}; stopping after epoch {epoch}.")
            break

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
    parser.add_argument("--weighted", action="store_true",
                        help="Use reward-weighted loss (for self-play data)")
    parser.add_argument("--v2", action="store_true",
                        help="Use V2 architecture (ResBlock + LayerNorm)")
    parser.add_argument("--soft-label", action="store_true",
                        help="Use EV soft-label loss (requires action_evs.npy)")
    parser.add_argument("--soft-temperature", type=float, default=2.0,
                        help="Temperature for soft-label softmax (default: 2.0)")
    parser.add_argument("--pretrained", default=None,
                        help="Path to pretrained BC policy to initialize from")
    parser.add_argument("--skip-vn", action="store_true",
                        help="Skip value network training (policy only)")
    parser.add_argument("--pretrained-vn", default=None,
                        help="Path to pretrained VN checkpoint to fine-tune from")
    parser.add_argument("--skip-policy", action="store_true",
                        help="Skip policy training (VN only)")
    parser.add_argument("--max-seconds", type=float, default=0.0,
                        help="Stop after approximately this many seconds (0 disables)")
    args = parser.parse_args()

    config = TrainingConfig(
        bc_lr=args.lr,
        bc_epochs=args.epochs,
        bc_batch_size=args.batch_size,
    )
    train_behavior_cloning(
        args.data, config, args.save, args.device, args.weighted, args.v2,
        soft_label=args.soft_label, soft_temperature=args.soft_temperature,
        pretrained=args.pretrained, pretrained_vn=args.pretrained_vn,
        skip_vn=args.skip_vn, skip_policy=args.skip_policy,
        max_seconds=args.max_seconds,
    )
