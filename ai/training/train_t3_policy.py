"""
T3 PolicyValueNet Training Script
===================================
Trains a dual-head model (Policy + Value) on 588K T3 game states.
- Policy head: predicts EV of each of the 27 possible actions (masked)
- Value head: predicts the best achievable EV from the state

Dataset: ai/data/t3_dataset_500k.npz
  - states:       (N, 490) float32
  - action_evs:   (N, 27)  float32
  - action_masks:  (N, 27)  bool
  - best_evs:     (N,)     float32
"""
import os
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class T3Dataset(Dataset):
    """Loads the merged NPZ file and serves (state, action_evs, mask, best_ev) tuples."""

    def __init__(self, npz_path: str):
        print(f"Loading dataset from {npz_path}...")
        t0 = time.time()
        data = np.load(npz_path)

        self.states = torch.tensor(data["states"], dtype=torch.float32)
        self.action_evs = torch.tensor(data["action_evs"], dtype=torch.float32)
        self.action_masks = torch.tensor(data["action_masks"], dtype=torch.bool)
        self.best_evs = torch.tensor(data["best_evs"], dtype=torch.float32)

        print(f"  Loaded {len(self)} states in {time.time()-t0:.1f}s")
        print(f"  State dim: {self.states.shape[1]}, Actions: {self.action_evs.shape[1]}")
        print(f"  Best EV range: [{self.best_evs.min():.2f}, {self.best_evs.max():.2f}]")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (self.states[idx], self.action_evs[idx],
                self.action_masks[idx], self.best_evs[idx])


# ──────────────────────────────────────────────
# Model Architecture
# ──────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class T3PolicyValueNet(nn.Module):
    """
    Dual-head architecture:
      - Shared trunk: input → ResBlocks
      - Policy head: predicts EV for each of 27 actions
      - Value head:  predicts scalar best EV
    """
    def __init__(self, input_dim=490, hidden_dim=1024, action_dim=27,
                 num_blocks=4, dropout=0.15):
        super().__init__()

        # Shared trunk
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])

        # Policy head (action EV prediction)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        # Value head (best EV prediction)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        policy = self.policy_head(x)  # (B, 27)
        value = self.value_head(x).squeeze(-1)   # (B,)
        return policy, value


# ──────────────────────────────────────────────
# Metrics (vectorized, fast)
# ──────────────────────────────────────────────
@torch.no_grad()
def compute_top1_accuracy(pred_evs, target_evs, masks):
    """Vectorized top-1 accuracy: did the model pick the best action?"""
    # Mask out invalid actions with -inf
    NEG_INF = -1e9
    masked_pred = pred_evs.clone()
    masked_targ = target_evs.clone()

    # Also filter un-evaluated actions (target == -1e9 sentinel)
    evaluated = masks & (target_evs > -1e8)

    masked_pred[~evaluated] = NEG_INF
    masked_targ[~evaluated] = NEG_INF

    # Only consider samples with at least 1 valid action
    has_valid = evaluated.any(dim=1)
    if not has_valid.any():
        return 0.0, 0

    pred_best = masked_pred[has_valid].argmax(dim=1)
    targ_best = masked_targ[has_valid].argmax(dim=1)
    correct = (pred_best == targ_best).sum().item()
    return correct, has_valid.sum().item()


@torch.no_grad()
def compute_ev_regret(pred_evs, target_evs, masks):
    """How much EV is lost by following the model's choice vs the optimal action."""
    NEG_INF = -1e9
    evaluated = masks & (target_evs > -1e8)

    masked_pred = pred_evs.clone()
    masked_pred[~evaluated] = NEG_INF

    masked_targ = target_evs.clone()
    masked_targ[~evaluated] = NEG_INF

    has_valid = evaluated.any(dim=1)
    if not has_valid.any():
        return 0.0, 0

    pred_choice = masked_pred[has_valid].argmax(dim=1)  # what model would pick
    # EV of model's choice according to ground truth
    chosen_ev = target_evs[has_valid].gather(1, pred_choice.unsqueeze(1)).squeeze(1)
    # EV of optimal choice
    optimal_ev = masked_targ[has_valid].max(dim=1).values

    regret = (optimal_ev - chosen_ev).mean().item()
    return regret, has_valid.sum().item()


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Paths
    base_dir = Path(__file__).resolve().parent.parent  # ai/
    data_path = base_dir / "data" / "t3_dataset_500k.npz"
    save_dir = base_dir / "data"
    save_dir.mkdir(exist_ok=True)
    model_path = save_dir / "t3_policyvalue_model.pt"
    history_path = save_dir / "t3_training_history.json"

    # Hyperparameters
    BATCH_SIZE = 2048
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    EPOCHS = 60
    PATIENCE = 10
    POLICY_WEIGHT = 1.0
    VALUE_WEIGHT = 0.5

    # Data
    dataset = T3Dataset(str(data_path))
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    print(f"Train: {train_size:,} | Val: {val_size:,}")

    # Model
    model = T3PolicyValueNet().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_loss = float('inf')
    patience_counter = 0
    history = []

    print(f"\n{'='*90}")
    print(f"{'Epoch':>5} | {'T-Policy':>9} {'T-Value':>8} {'T-Acc':>7} | "
          f"{'V-Policy':>9} {'V-Value':>8} {'V-Acc':>7} {'V-Regret':>9} | {'LR':>10}")
    print(f"{'='*90}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_policy_loss = 0.0
        train_value_loss = 0.0
        train_correct = 0
        train_total = 0
        n_batches = 0

        for states, action_evs, masks, best_evs in train_loader:
            states = states.to(device)
            action_evs = action_evs.to(device)
            masks = masks.to(device)
            best_evs = best_evs.to(device)

            optimizer.zero_grad()
            pred_policy, pred_value = model(states)

            # Policy loss: MSE on valid+evaluated actions only
            evaluated = masks & (action_evs > -1e8)
            valid_preds = pred_policy[evaluated]
            valid_targets = action_evs[evaluated]

            if len(valid_preds) > 0:
                p_loss = F.mse_loss(valid_preds, valid_targets)
            else:
                p_loss = torch.tensor(0.0, device=device)

            # Value loss: MSE on best_ev
            v_loss = F.mse_loss(pred_value, best_evs)

            # Combined loss
            loss = POLICY_WEIGHT * p_loss + VALUE_WEIGHT * v_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_policy_loss += p_loss.item()
            train_value_loss += v_loss.item()

            # Accuracy
            c, t = compute_top1_accuracy(pred_policy, action_evs, masks)
            train_correct += c
            train_total += t
            n_batches += 1

        scheduler.step()

        avg_tp = train_policy_loss / n_batches
        avg_tv = train_value_loss / n_batches
        train_acc = train_correct / max(1, train_total)

        # ── Validation ──
        model.eval()
        val_policy_loss = 0.0
        val_value_loss = 0.0
        val_correct = 0
        val_total = 0
        val_regret_sum = 0.0
        val_regret_n = 0
        n_val_batches = 0

        with torch.no_grad():
            for states, action_evs, masks, best_evs in val_loader:
                states = states.to(device)
                action_evs = action_evs.to(device)
                masks = masks.to(device)
                best_evs = best_evs.to(device)

                pred_policy, pred_value = model(states)

                evaluated = masks & (action_evs > -1e8)
                valid_preds = pred_policy[evaluated]
                valid_targets = action_evs[evaluated]

                if len(valid_preds) > 0:
                    val_policy_loss += F.mse_loss(valid_preds, valid_targets).item()
                val_value_loss += F.mse_loss(pred_value, best_evs).item()

                c, t = compute_top1_accuracy(pred_policy, action_evs, masks)
                val_correct += c
                val_total += t

                r, rn = compute_ev_regret(pred_policy, action_evs, masks)
                val_regret_sum += r * rn
                val_regret_n += rn
                n_val_batches += 1

        avg_vp = val_policy_loss / n_val_batches
        avg_vv = val_value_loss / n_val_batches
        val_acc = val_correct / max(1, val_total)
        val_regret = val_regret_sum / max(1, val_regret_n)
        current_lr = optimizer.param_groups[0]['lr']

        elapsed = time.time() - t0
        combined_val = POLICY_WEIGHT * avg_vp + VALUE_WEIGHT * avg_vv

        # Logging
        print(f"{epoch:5d} | {avg_tp:9.4f} {avg_tv:8.4f} {train_acc:7.2%} | "
              f"{avg_vp:9.4f} {avg_vv:8.4f} {val_acc:7.2%} {val_regret:9.4f} | "
              f"{current_lr:.2e} ({elapsed:.1f}s)")

        history.append({
            "epoch": epoch,
            "train_policy_mse": avg_tp, "train_value_mse": avg_tv, "train_acc": train_acc,
            "val_policy_mse": avg_vp, "val_value_mse": avg_vv, "val_acc": val_acc,
            "val_regret": val_regret, "lr": current_lr
        })

        # Early stopping / best model save
        if combined_val < best_val_loss:
            best_val_loss = combined_val
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_policy_mse': avg_vp,
                'val_value_mse': avg_vv,
                'val_acc': val_acc,
                'val_regret': val_regret,
            }, str(model_path))
            print(f"       ★ Best model saved (combined val loss: {combined_val:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    # Save history
    with open(str(history_path), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*90}")
    print(f"Training complete!")
    print(f"  Best model: {model_path}")
    print(f"  History:    {history_path}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"{'='*90}")


if __name__ == "__main__":
    train()
