"""
OFC Pineapple - Ranking-based Behavior Cloning for T0 PolicyNet v2

Uses ListNet ranking loss with ONLINE suit augmentation (24x data).
Suit permutation on state tensor preserves action indices (verified).

Usage:
    python ai/training/train_ranking.py --data ai/data/ranked_t0 --epochs 300
"""
import sys
import argparse
import json
import random
import itertools
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.models.networks import PolicyNetworkV2, _adapt_state
from ai.engine.encoding import STATE_DIM
from ai.engine.action_space import MAX_ACTIONS


# ─── Suit Augmentation on Tensors ───────────────────────────────────────
# State layout: 54 cards × 9 locations = 486 dims
# Cards ordered: h(0-12), d(13-25), c(26-38), s(39-51), jokers(52-53)
# Each card has 9 location features → h block = indices 0:117, d = 117:234, etc.

SUIT_BLOCK_SIZE = 13 * 9  # 117
SUIT_PERMS_24 = list(itertools.permutations([0, 1, 2, 3]))  # 24 suit permutations


def augment_state_suits(state_batch: torch.Tensor, perm_indices: list = None):
    """Apply random suit permutation to a batch of state tensors.
    
    Swaps the 4 suit blocks (h,d,c,s) in the card portion of the state.
    Leaves joker features and meta features unchanged.
    
    Args:
        state_batch: (batch, 522) tensor
        perm_indices: list of 4 ints, e.g. [2,0,3,1] means h→c, d→h, c→s, s→d
    
    Returns:
        Augmented state tensor (same shape)
    """
    if perm_indices is None:
        perm_indices = list(random.choice(SUIT_PERMS_24))
    
    # Identity permutation → no-op
    if perm_indices == [0, 1, 2, 3]:
        return state_batch
    
    result = state_batch.clone()
    
    # Permute suit blocks in card section
    for new_pos, old_pos in enumerate(perm_indices):
        src_start = old_pos * SUIT_BLOCK_SIZE
        src_end = src_start + SUIT_BLOCK_SIZE
        dst_start = new_pos * SUIT_BLOCK_SIZE
        dst_end = dst_start + SUIT_BLOCK_SIZE
        result[:, dst_start:dst_end] = state_batch[:, src_start:src_end]
    
    # Jokers (468:486) and meta features (486:522) stay unchanged
    return result


# ─── Dataset ────────────────────────────────────────────────────────────

class RankingDataset:
    """Ranking dataset with online suit augmentation."""

    def __init__(self, data_dir: str, device: str = "cpu"):
        self.device = torch.device(device)
        d = Path(data_dir)

        self.states = torch.from_numpy(
            np.load(d / "states.npy").astype(np.float32)
        ).to(self.device)
        self.action_evs = torch.from_numpy(
            np.load(d / "action_evs.npy").astype(np.float32)
        ).to(self.device)
        self.valid_masks = torch.from_numpy(
            np.load(d / "valid_masks.npy")
        ).bool().to(self.device)
        self.best_actions = torch.from_numpy(
            np.load(d / "actions.npy")
        ).long().to(self.device)

        self.input_dim = self.states.shape[1]
        print(f"  Loaded {len(self.states)} hands, input_dim={self.input_dim}")

    def __len__(self):
        return len(self.states)


# ─── Model ──────────────────────────────────────────────────────────────

class PolicyNetSmall(nn.Module):
    """~500K param PolicyNet for better generalization."""

    def __init__(self, input_dim: int = STATE_DIM,
                 max_actions: int = MAX_ACTIONS,
                 hidden: int = 256, n_blocks: int = 2,
                 dropout: float = 0.15):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, hidden),
        )
        self.blocks = nn.ModuleList([
            ResBlockSmall(hidden, dropout) for _ in range(n_blocks)
        ])
        self.output = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, max_actions),
        )

    def forward_logits(self, state, valid_mask):
        state = _adapt_state(state, self.input_proj[0].in_features)
        x = self.input_proj(state)
        for block in self.blocks:
            x = block(x)
        logits = self.output(x)
        logits = logits.masked_fill(~valid_mask, float('-inf'))
        return logits

    def forward(self, state, valid_mask):
        return F.softmax(self.forward_logits(state, valid_mask), dim=-1)

    def select_action(self, state, valid_mask, temperature=1.0):
        with torch.no_grad():
            logits = self.forward_logits(state, valid_mask)
            if temperature == 0:
                return torch.argmax(logits, dim=-1)
            scaled = logits / max(temperature, 1e-8)
            probs = F.softmax(scaled, dim=-1)
            return torch.multinomial(probs, 1).squeeze(-1)


class ResBlockSmall(nn.Module):
    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


# ─── Loss Functions ─────────────────────────────────────────────────────

def listnet_loss(logits, target_evs, valid_mask, temperature=2.0):
    """ListNet: KL(softmax(EV/T) || softmax(logits))"""
    has_ev = (target_evs > -1e8) & valid_mask
    
    # Safety: skip samples with no valid EVs
    n_valid = has_ev.float().sum(dim=-1)
    sample_mask = n_valid > 1  # need at least 2 actions to rank
    if not sample_mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    masked_logits = logits.masked_fill(~has_ev, -1e9)
    masked_evs = target_evs.masked_fill(~has_ev, -1e9)
    
    target_probs = F.softmax(masked_evs / temperature, dim=-1)
    log_model_probs = F.log_softmax(masked_logits, dim=-1)
    
    kl = target_probs * (torch.log(target_probs.clamp(min=1e-10)) - log_model_probs)
    kl = kl.masked_fill(~has_ev, 0.0)
    
    per_sample = kl.sum(dim=-1)
    return (per_sample * sample_mask.float()).sum() / sample_mask.float().sum().clamp(min=1)


def ev_regression_loss(logits, target_evs, valid_mask):
    """Direct EV regression: MSE between predicted scores and scaled EVs."""
    has_ev = (target_evs > -1e8) & valid_mask
    n_valid = has_ev.float().sum(dim=-1).clamp(min=1)
    
    # Scale EVs to roughly [-3, 3] range
    ev_safe = target_evs.clone()
    ev_safe[~has_ev] = 0
    ev_scaled = ev_safe / 20.0  # typical EVs range 0-50, so /20 → 0-2.5
    
    # Replace -inf logits with 0 before computing MSE (they'll be masked anyway)
    logits_safe = logits.clone()
    logits_safe[~has_ev] = 0
    
    diff = (logits_safe - ev_scaled).pow(2) * has_ev.float()
    return (diff.sum(dim=-1) / n_valid).mean()


def combined_loss(logits, target_evs, valid_mask, temperature=2.0, alpha=0.7):
    """Combine ranking + regression for stability."""
    l_rank = listnet_loss(logits, target_evs, valid_mask, temperature)
    l_reg = ev_regression_loss(logits, target_evs, valid_mask)
    return alpha * l_rank + (1 - alpha) * l_reg


# ─── Training ───────────────────────────────────────────────────────────

def train_ranking(
    data_dir: str,
    save_dir: str = None,
    epochs: int = 300,
    batch_size: int = 32,
    lr: float = 5e-4,
    weight_decay: float = 1e-3,
    temperature: float = 2.0,
    loss_type: str = "combined",
    device: str = "auto",
    augment: bool = True,
    hidden: int = 256,
    n_blocks: int = 2,
    dropout: float = 0.15,
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    ds = RankingDataset(data_dir, device=device)
    n_total = len(ds)
    n_train = int(0.85 * n_total)
    n_val = n_total - n_train
    
    # Fix random seed for reproducible split
    gen = torch.Generator().manual_seed(42)
    perm = torch.randperm(n_total, generator=gen, device="cpu")
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    effective_train = n_train * (24 if augment else 1)
    print(f"Data: {n_total} hands (train={n_train}, val={n_val})")
    print(f"Online suit augmentation: {'24x' if augment else 'off'} → {effective_train} effective samples")

    # Model
    model = PolicyNetSmall(
        input_dim=ds.input_dim,
        max_actions=MAX_ACTIONS,
        hidden=hidden,
        n_blocks=n_blocks,
        dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: PolicyNetSmall ({n_params:,} params)")
    print(f"Loss: {loss_type}, temperature={temperature}")
    print(f"Overparameterization ratio: {n_params / effective_train:.1f}x")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-5
    )

    if save_dir is None:
        save_dir = "ai/models/t0_ranking_v2"
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    best_val_top1 = 0.0
    best_epoch = 0
    patience = 0
    max_patience = 80
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_top1 = 0
        train_top5 = 0
        train_total = 0

        # Each epoch: iterate through training data with random suit augmentation
        shuffled = train_idx[torch.randperm(len(train_idx))]

        for i in range(0, len(shuffled), batch_size):
            idx = shuffled[i:i+batch_size]
            states = ds.states[idx]
            action_evs = ds.action_evs[idx]
            valid_mask = ds.valid_masks[idx]
            best_action = ds.best_actions[idx]

            # Online suit augmentation
            if augment:
                perm_idx = list(random.choice(SUIT_PERMS_24))
                states = augment_state_suits(states, perm_idx)

            logits = model.forward_logits(states, valid_mask)

            if loss_type == "listnet":
                loss = listnet_loss(logits, action_evs, valid_mask, temperature)
            elif loss_type == "regression":
                loss = ev_regression_loss(logits, action_evs, valid_mask)
            else:
                loss = combined_loss(logits, action_evs, valid_mask, temperature)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if not torch.isnan(loss):
                train_loss += loss.item() * len(states)

            with torch.no_grad():
                preds = logits.argsort(dim=-1, descending=True)
                for j in range(len(states)):
                    ba = best_action[j].item()
                    ranking = preds[j].tolist()
                    if ba in ranking[:1]:
                        train_top1 += 1
                    if ba in ranking[:5]:
                        train_top5 += 1
                train_total += len(states)

        scheduler.step()

        # Validation (no augmentation, measure generalization)
        model.eval()
        val_loss = 0.0
        val_top1 = 0
        val_top5 = 0
        val_top10 = 0
        val_top50 = 0
        val_total = 0
        val_ev_gap = 0.0

        with torch.no_grad():
            for i in range(0, len(val_idx), batch_size):
                idx = val_idx[i:i+batch_size]
                states = ds.states[idx]
                action_evs = ds.action_evs[idx]
                valid_mask = ds.valid_masks[idx]
                best_action = ds.best_actions[idx]

                logits = model.forward_logits(states, valid_mask)

                if loss_type == "listnet":
                    loss = listnet_loss(logits, action_evs, valid_mask, temperature)
                elif loss_type == "regression":
                    loss = ev_regression_loss(logits, action_evs, valid_mask)
                else:
                    loss = combined_loss(logits, action_evs, valid_mask, temperature)

                loss_val = loss.item()
                if not (torch.isnan(loss) or loss_val != loss_val):
                    val_loss += loss_val * len(states)

                preds = logits.argsort(dim=-1, descending=True)
                for j in range(len(states)):
                    ba = best_action[j].item()
                    ranking = preds[j].tolist()
                    pred_best = ranking[0]

                    if ba in ranking[:1]:
                        val_top1 += 1
                    if ba in ranking[:5]:
                        val_top5 += 1
                    if ba in ranking[:10]:
                        val_top10 += 1
                    if ba in ranking[:50]:
                        val_top50 += 1

                    best_ev = action_evs[j, ba].item()
                    pred_ev = action_evs[j, pred_best].item()
                    if pred_ev > -1e8:
                        val_ev_gap += (best_ev - pred_ev)

                val_total += len(states)

        train_loss_avg = train_loss / max(train_total, 1)
        val_loss_avg = val_loss / max(val_total, 1)
        val_ev_gap_avg = val_ev_gap / max(val_total, 1)
        val_top1_pct = val_top1 / max(val_total, 1)

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss_avg,
            "val_loss": val_loss_avg,
            "train_top1": train_top1 / max(train_total, 1),
            "train_top5": train_top5 / max(train_total, 1),
            "val_top1": val_top1_pct,
            "val_top5": val_top5 / max(val_total, 1),
            "val_top10": val_top10 / max(val_total, 1),
            "val_top50": val_top50 / max(val_total, 1),
            "val_ev_gap": val_ev_gap_avg,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Ep {epoch+1:3d}/{epochs} | "
                f"Loss t={train_loss_avg:.4f} v={val_loss_avg:.4f} | "
                f"Top1 t={record['train_top1']:.1%} v={val_top1_pct:.1%} | "
                f"Top5 v={record['val_top5']:.1%} | "
                f"Top10 v={record['val_top10']:.1%} | "
                f"Top50 v={record['val_top50']:.1%} | "
                f"EVgap={val_ev_gap_avg:.2f}"
            )

        # Save best by val_loss
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_epoch = epoch + 1
            patience = 0
            torch.save(model.state_dict(), save_path / "bc_policy_best.pt")
        else:
            patience += 1

        # Also track best top1
        if val_top1_pct > best_val_top1:
            best_val_top1 = val_top1_pct
            torch.save(model.state_dict(), save_path / "bc_policy_best_top1.pt")

        if patience >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1} (patience={max_patience})")
            break

    torch.save(model.state_dict(), save_path / "bc_policy_final.pt")

    with open(save_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save config
    config = {
        "model": "PolicyNetSmall",
        "input_dim": ds.input_dim,
        "max_actions": MAX_ACTIONS,
        "hidden": 256,
        "n_blocks": 2,
        "n_params": n_params,
        "loss": loss_type,
        "temperature": temperature,
        "augment": augment,
        "n_train": n_train,
        "n_val": n_val,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }
    with open(save_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    final = history[-1]
    print(f"  Final val metrics:")
    print(f"    Top-1:  {final['val_top1']:.1%}")
    print(f"    Top-5:  {final['val_top5']:.1%}")
    print(f"    Top-10: {final['val_top10']:.1%}")
    print(f"    Top-50: {final['val_top50']:.1%}")
    print(f"    EV gap: {final['val_ev_gap']:.2f}")
    print(f"  Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ai/data/ranked_t0")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--loss", choices=["listnet", "regression", "combined"], default="combined")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    args = parser.parse_args()

    train_ranking(
        data_dir=args.data,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        loss_type=args.loss,
        augment=not args.no_augment,
        hidden=args.hidden,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
    )
