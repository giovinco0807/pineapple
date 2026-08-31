"""
T3 Oracle Training — PolicyValueNet on GCP-generated NPZ dataset

Data format (from GCP Rust solver):
  states:      (N, 490) float32
  action_evs:  (N, 27)  float32  [EV per action, -inf for invalid]
  action_masks:(N, 27)  bool
  best_evs:    (N,)     float32  [best achievable EV]

Model: T2PolicyValueNet
  Input: 490-dim state
  Policy head: 27 logits (ListNet ranking loss vs action_evs)
  Value head:  1 scalar (MSE vs best_evs)

Usage:
  python ai/training/train_t2_oracle.py --data-dir ai/data/t2_oracle --epochs 80
"""
import sys
import argparse
import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1  = nn.Linear(dim, dim)
        self.fc2  = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        r = x
        x = self.norm(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x + r


class T2PolicyValueNet(nn.Module):
    """
    9.7M parameter PolicyValueNet matching the GCP-trained architecture.

    Input:  490-dim state
    Policy: 27 logits (action ranking)
    Value:  1 scalar (best EV prediction)
    """

    def __init__(self, state_dim: int = 490, n_actions: int = 27,
                 hidden: int = 1024, n_blocks: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList([
            ResBlock(hidden, dropout) for _ in range(n_blocks)
        ])
        self.trunk_out = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 512),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(512, n_actions)
        self.value_head  = nn.Linear(512, 1)

    def forward(self, states, masks=None):
        x = self.input_proj(states)
        for block in self.blocks:
            x = block(x)
        x = self.trunk_out(x)
        logits = self.policy_head(x)
        if masks is not None:
            logits = logits.masked_fill(~masks, float('-inf'))
        value = self.value_head(x).squeeze(-1)
        return logits, value


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class T2Dataset:
    """Load all NPZ files from t2_oracle sub-dirs into GPU tensors."""

    def __init__(self, data_dirs, device='cpu', verbose=True):
        all_states, all_evs, all_masks, all_best = [], [], [], []

        for d in data_dirs:
            d = Path(d)
            files = sorted(d.glob('*.npz'))
            if not files:
                continue
            for f in files:
                try:
                    npz = np.load(f)
                    all_states.append(npz['states'])        # (300, 490)
                    all_evs.append(npz['action_evs'])       # (300, 27)
                    all_masks.append(npz['action_masks'])   # (300, 27)
                    all_best.append(npz['best_evs'])        # (300,)
                except Exception as e:
                    if verbose:
                        print(f"  Warning: skip {f.name}: {e}")

        states = np.concatenate(all_states, axis=0).astype(np.float32)
        evs    = np.concatenate(all_evs,    axis=0).astype(np.float32)
        masks  = np.concatenate(all_masks,  axis=0)
        bests  = np.concatenate(all_best,   axis=0).astype(np.float32)

        self.states = torch.from_numpy(states).to(device)
        self.evs    = torch.from_numpy(evs).to(device)
        self.masks  = torch.from_numpy(masks).to(device)
        self.bests  = torch.from_numpy(bests).to(device)

        if verbose:
            print(f"  Loaded {len(self.states):,} samples from {len(data_dirs)} dirs")
            print(f"  State dim: {self.states.shape[1]}")
            valid_evs = evs[evs > -1e8]
            print(f"  EV range: [{valid_evs.min():.2f}, {valid_evs.max():.2f}]")
            print(f"  Best EV:  mean={bests.mean():.2f} ± {bests.std():.2f}")
            print(f"  Avg valid actions: {masks.sum(1).mean():.1f}")

    def __len__(self):
        return len(self.states)


# ─────────────────────────────────────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────────────────────────────────────

def listnet_loss(logits, target_evs, valid_mask, temperature=3.0):
    """ListNet: KL(softmax(EV/T) || softmax(logits))"""
    has_ev = (target_evs > -1e8) & valid_mask
    n_valid = has_ev.float().sum(dim=-1)
    sample_mask = n_valid > 1
    if not sample_mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    masked_logits = logits.masked_fill(~has_ev, -1e9)
    masked_evs    = target_evs.masked_fill(~has_ev, -1e9)

    target_probs    = F.softmax(masked_evs / temperature, dim=-1)
    log_model_probs = F.log_softmax(masked_logits, dim=-1)

    kl = target_probs * (torch.log(target_probs.clamp(min=1e-10)) - log_model_probs)
    kl = kl.masked_fill(~has_ev, 0.0)

    per_sample = kl.sum(dim=-1)
    return (per_sample * sample_mask.float()).sum() / sample_mask.float().sum().clamp(min=1)


def ev_regret(logits, action_evs, masks):
    """Mean EV regret: best_ev - predicted_best_ev."""
    has_ev = (action_evs > -1e8) & masks
    pred_idx = logits.masked_fill(~has_ev, -1e9).argmax(dim=-1)
    pred_ev  = action_evs.gather(1, pred_idx.unsqueeze(1)).squeeze(1)
    best_ev  = action_evs.masked_fill(~has_ev, -1e9).max(dim=-1).values
    valid = has_ev.any(dim=-1)
    regret = (best_ev - pred_ev)[valid]
    return regret.mean().item() if len(regret) > 0 else 0.0


def top1_accuracy(logits, action_evs, masks):
    """% of samples where predicted top action == true best action."""
    has_ev = (action_evs > -1e8) & masks
    pred_best = logits.masked_fill(~has_ev, -1e9).argmax(dim=-1)
    true_best = action_evs.masked_fill(~has_ev, -1e9).argmax(dim=-1)
    valid = has_ev.any(dim=-1)
    correct = (pred_best[valid] == true_best[valid]).float()
    return correct.mean().item() if len(correct) > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train_t2_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== T2 Oracle Training ===")
    print(f"Device: {device}")
    
    # Check if data dir exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data dir not found: {args.data_dir}")
        return
        
    print(f"Loading data from: {data_dir}")
    dataset_device = device
    if args.dataset_device != "auto":
        dataset_device = torch.device(args.dataset_device)
    print(f"Dataset tensors: {dataset_device}")
    ds = T2Dataset([str(data_dir)], device=dataset_device)
    N = len(ds)

    # ── Train/Val split ──
    gen = torch.Generator()
    gen.manual_seed(42)
    perm = torch.randperm(N, generator=gen)
    n_val   = min(N - 1, max(1, int(N * 0.2))) if N < 2000 else max(1000, int(N * 0.1))
    n_train = N - n_val
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]
    print(f"  Train: {n_train:,}  Val: {n_val:,}")

    # ── Model ──
    model = T2PolicyValueNet(
        state_dim=ds.states.shape[1],
        n_actions=ds.evs.shape[1],
        hidden=args.hidden,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: T2PolicyValueNet ({n_params:,} params)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_regret = float('inf')
    best_epoch = 0
    history = []
    start_epoch = 1

    if args.init_model and not args.resume:
        init_path = Path(args.init_model)
        print(f"  Initializing from {init_path}...")
        ckpt = torch.load(init_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)

    if args.resume:
        model_path = out_dir / 't2_policyvalue_model.pt'
        hist_path = out_dir / 't2_training_history.json'
        if model_path.exists():
            print(f"  Resuming from {model_path}...")
            ckpt = torch.load(model_path, map_location=device)
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
                start_epoch = ckpt.get('epoch', 0) + 1
                best_val_regret = ckpt.get('val_regret', float('inf'))
            else:
                model.load_state_dict(ckpt)
        if hist_path.exists():
            with open(hist_path, 'r') as f:
                history = json.load(f)

    t0 = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        # Shuffle train indices
        perm_e = train_idx[torch.randperm(len(train_idx))]
        train_policy_loss = 0.0
        train_value_loss  = 0.0
        n_batches = 0

        for i in range(0, len(perm_e), args.batch_size):
            idx = perm_e[i:i + args.batch_size]
            states = ds.states[idx].to(device, non_blocking=True)
            evs    = ds.evs[idx].to(device, non_blocking=True)
            masks  = ds.masks[idx].to(device, non_blocking=True)
            bests  = ds.bests[idx].to(device, non_blocking=True)

            logits, value = model(states, masks)

            # Policy loss: ListNet ranking
            p_loss = listnet_loss(logits, evs, masks, args.temperature)
            # Value loss: MSE vs best EV
            v_loss = F.mse_loss(value, bests)

            loss = p_loss + args.value_weight * v_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_policy_loss += p_loss.item()
            train_value_loss  += v_loss.item()
            n_batches += 1

        scheduler.step()

        # ── Validation ──
        if epoch % args.eval_every == 0 or epoch == 1 or epoch == args.epochs:
            model.eval()
            val_policy_loss = 0.0
            val_regret      = 0.0
            val_top1        = 0.0
            val_n_batches   = 0

            with torch.no_grad():
                for i in range(0, len(val_idx), args.batch_size):
                    idx = val_idx[i:i + args.batch_size]
                    states = ds.states[idx].to(device, non_blocking=True)
                    evs    = ds.evs[idx].to(device, non_blocking=True)
                    masks  = ds.masks[idx].to(device, non_blocking=True)
                    bests  = ds.bests[idx].to(device, non_blocking=True)

                    logits, value = model(states, masks)
                    val_policy_loss += listnet_loss(logits, evs, masks, args.temperature).item()
                    val_regret      += ev_regret(logits, evs, masks)
                    val_top1        += top1_accuracy(logits, evs, masks)
                    val_n_batches   += 1

            val_policy_loss /= max(val_n_batches, 1)
            val_regret      /= max(val_n_batches, 1)
            val_top1        /= max(val_n_batches, 1)
            elapsed          = time.time() - t0
            lr_now           = optimizer.param_groups[0]['lr']

            record = {
                'epoch': epoch,
                'train_policy_loss': train_policy_loss / max(n_batches, 1),
                'train_value_loss':  train_value_loss  / max(n_batches, 1),
                'val_policy_loss':   val_policy_loss,
                'val_regret':        val_regret,
                'val_top1':          val_top1,
                'lr': lr_now,
                'elapsed': elapsed,
            }
            history.append(record)

            marker = ''
            if val_regret < best_val_regret:
                best_val_regret = val_regret
                best_epoch = epoch
                marker = ' ★'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_regret': val_regret,
                    'val_top1':   val_top1,
                    'state_dim':  ds.states.shape[1],
                    'n_actions':  ds.evs.shape[1],
                    'hidden':     args.hidden,
                    'n_blocks':   args.n_blocks,
                }, out_dir / 't2_policyvalue_model.pt')

            print(
                f"  [{epoch:3d}/{args.epochs}] "
                f"p_loss={record['train_policy_loss']:.4f} "
                f"v_loss={record['train_value_loss']:.4f} | "
                f"val_regret={val_regret:.3f} "
                f"top1={val_top1:.1%} "
                f"lr={lr_now:.1e} "
                f"({elapsed:.0f}s){marker}"
            )

    # ── Final save ──
    torch.save(model.state_dict(), out_dir / 't2_policyvalue_final.pt')

    with open(out_dir / 't2_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training Complete")
    print(f"{'='*60}")
    print(f"  Best val regret: {best_val_regret:.3f} at epoch {best_epoch}")
    print(f"  Model saved: {out_dir / 't2_policyvalue_model.pt'}")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train T2 Oracle PolicyValueNet')
    parser.add_argument("--data-dir", type=str, default="ai/data/t2_oracle", help="Directory with NPZ files")
    parser.add_argument("--out-dir", type=str, default="ai/data/t2_oracle", help="Output directory for weights")
    parser.add_argument('--epochs',       type=int,   default=80)
    parser.add_argument('--batch-size',   type=int,   default=2048)
    parser.add_argument('--lr',           type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--hidden',       type=int,   default=1024)
    parser.add_argument('--n-blocks',     type=int,   default=4)
    parser.add_argument('--dropout',      type=float, default=0.1)
    parser.add_argument('--temperature',  type=float, default=3.0,
                        help='ListNet softmax temperature')
    parser.add_argument('--value-weight', type=float, default=0.3,
                        help='Weight for value head MSE loss')
    parser.add_argument('--eval-every',   type=int,   default=5,
                        help='Evaluate every N epochs')
    parser.add_argument('--device',       default='auto')
    parser.add_argument('--dataset-device', default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Where to keep the loaded tensors. Use cpu for multi-million sample datasets.')
    parser.add_argument('--resume',       action='store_true',
                        help='Resume training from checkpoint in save-dir')
    parser.add_argument('--init-model',   default=None,
                        help='Optional checkpoint to initialize weights before a fresh run')
    args = parser.parse_args()

    train_t2_model(args)
