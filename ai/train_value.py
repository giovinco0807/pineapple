"""
Value Network Training (Multi-task)

Trains ValueNetwork on (obs_520, score, bust, fl) tuples.
  - value_head: MSE on normalized game score
  - bust_head:  BCE on bust label (if available)
  - fl_head:    BCE on FL entry label (if available)

Usage:
    python ai/train_value.py --data data/value_data_v5.npz --epochs 200 --save ai/models/value_v5
    python ai/train_value.py --data data/value_data_v5.npz --multitask --bust-weight 0.3 --fl-weight-loss 0.3

Architecture:
    Input: 520-dim state vector
    Shared: 520 → 1024 → 512 → 256 (ReLU)
    Heads: value (1), bust_prob (1, sigmoid), fl_prob (1, sigmoid)
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler

from ai.models.networks import ValueNetwork, ValueNetworkV2, ValueNetworkV3


class ValueDataset(Dataset):
    """Dataset for Value Network training (multi-task).

    Supports loading from multiple NPZ files (e.g., on-policy + off-policy).
    Uses float16 storage for obs to save RAM (~50%), converts to float32 per batch.
    """

    def __init__(self, npz_paths, normalize_scores: bool = True, max_samples_per_file: int = 0,
                 onpolicy_index: int = 0):
        """
        Args:
            npz_paths: List of NPZ file paths
            onpolicy_index: Index of the on-policy file (default: 0 = first file).
                           Samples from this file get is_onpolicy=True.
        """
        if isinstance(npz_paths, (str, Path)):
            npz_paths = [npz_paths]

        obs_parts = []
        score_parts = []
        turn_parts = []
        bust_parts = []
        fl_parts = []
        onpolicy_parts = []

        for file_idx, path in enumerate(npz_paths):
            data = np.load(path)
            n_total = len(data['score'])

            # Subsample large files to avoid OOM
            if max_samples_per_file > 0 and n_total > max_samples_per_file:
                idx = np.random.choice(n_total, max_samples_per_file, replace=False)
                idx.sort()
                obs_raw = data['obs'][idx]
                score_raw = data['score'][idx]
                turn_raw = data['turn'][idx]
                bust_raw = data['busted'][idx] if 'busted' in data else None
                fl_raw = None
                if 'busted' in data:
                    if 'fl_entry' in data:
                        fl_raw = data['fl_entry'][idx]
                    elif 'fl' in data:
                        fl_raw = data['fl'][idx]
                print(f"  Loaded {max_samples_per_file:,}/{n_total:,} samples from {path} (subsampled)")
            else:
                obs_raw = data['obs']
                score_raw = data['score']
                turn_raw = data['turn']
                bust_raw = data['busted'] if 'busted' in data else None
                # Support both 'fl_entry' and 'fl' keys
                fl_raw = None
                if 'busted' in data:
                    if 'fl_entry' in data:
                        fl_raw = data['fl_entry']
                    elif 'fl' in data:
                        fl_raw = data['fl']
                print(f"  Loaded {n_total:,} samples from {path}")

            obs_parts.append(obs_raw.astype(np.float16))
            score_parts.append(score_raw.astype(np.float32))
            turn_parts.append(turn_raw.astype(np.int64))
            if bust_raw is not None:
                bust_parts.append(bust_raw.astype(np.float32))
                fl_parts.append(fl_raw.astype(np.float32))
            else:
                bust_parts.append((score_raw < -3).astype(np.float32))
                fl_parts.append((score_raw > 15).astype(np.float32))
            # Track on-policy vs off-policy
            n_loaded = len(obs_raw)
            onpolicy_parts.append(np.full(n_loaded, file_idx == onpolicy_index, dtype=np.bool_))
            del data  # Free NPZ memory

        # Concatenate all sources (float16 obs to save RAM)
        self.obs = torch.from_numpy(np.concatenate(obs_parts))       # (N, 520) float16
        del obs_parts
        self.scores = torch.FloatTensor(np.concatenate(score_parts))  # (N,)
        self.turns = torch.LongTensor(np.concatenate(turn_parts))     # (N,)
        self.busted = torch.FloatTensor(np.concatenate(bust_parts))
        self.fl_entry = torch.FloatTensor(np.concatenate(fl_parts))
        self.is_onpolicy = torch.BoolTensor(np.concatenate(onpolicy_parts))
        self.has_labels = True

        self.score_mean = self.scores.mean().item()
        self.score_std = self.scores.std().item()

        if normalize_scores and self.score_std > 0:
            self.targets = (self.scores - self.score_mean) / self.score_std
        else:
            self.targets = self.scores

        # Stats
        print(f"  Total: {len(self):,} samples")
        print(f"  Score stats: mean={self.score_mean:.2f}, std={self.score_std:.2f}")
        print(f"  Score range: [{self.scores.min():.1f}, {self.scores.max():.1f}]")
        n_b = self.busted.sum().item()
        n_f = self.fl_entry.sum().item()
        print(f"  Bust labels: {int(n_b):,} ({n_b/len(self)*100:.1f}%)")
        print(f"  FL labels:   {int(n_f):,} ({n_f/len(self)*100:.1f}%)")
        print(f"  Turn distribution: {torch.bincount(self.turns).tolist()}")
        n_on = self.is_onpolicy.sum().item()
        print(f"  On-policy: {n_on:,} ({n_on/len(self)*100:.1f}%), Off-policy: {len(self)-n_on:,}")

    def get_sample_weights(self, fl_threshold=15.0, fl_weight=3.0):
        """Compute per-sample weights for FL oversampling."""
        weights = torch.ones(len(self))
        fl_mask = self.scores > fl_threshold
        weights[fl_mask] = fl_weight
        n_fl = fl_mask.sum().item()
        print(f"  FL oversampling: {n_fl} samples x{fl_weight} (threshold={fl_threshold})")
        return weights

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return (self.obs[idx].float(), self.targets[idx], self.turns[idx],
                self.busted[idx], self.fl_entry[idx])


def train_epoch(model, loader, optimizer, criterion, device,
                bust_weight=0.0, fl_weight_loss=0.0, use_turn=False):
    """Train one epoch with optional multi-task loss."""
    model.train()
    total_loss = 0
    n_batches = 0
    bce = nn.BCELoss()

    for obs, target, turns, busted, fl_entry in loader:
        obs = obs.to(device)
        target = target.to(device)

        if use_turn:
            output = model(obs, turns.to(device))
        else:
            output = model(obs)
        pred = output['value'].squeeze(-1)
        loss = criterion(pred, target)

        # Multi-task: bust_prob BCE
        if bust_weight > 0:
            bust_pred = output['bust_prob'].squeeze(-1)
            loss = loss + bust_weight * bce(bust_pred, busted.to(device))

        # Multi-task: fl_prob BCE
        if fl_weight_loss > 0:
            fl_pred = output['fl_prob'].squeeze(-1)
            loss = loss + fl_weight_loss * bce(fl_pred, fl_entry.to(device))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train_epoch_gpu(obs_gpu, targets_gpu, turns_gpu, busted_gpu, fl_gpu,
                    indices, model, optimizer, criterion, batch_size,
                    bust_weight=0.0, fl_weight_loss=0.0, use_turn=False,
                    onpolicy_gpu=None, value_onpolicy_only=False):
    """Train one epoch with GPU-preloaded tensors and manual batching.

    Much faster than DataLoader: no CPU→GPU transfer, no sampler overhead.

    If value_onpolicy_only=True and onpolicy_gpu is provided:
      - value loss: computed only on on-policy samples
      - bust/FL loss: computed on ALL samples
    This prevents off-policy data from distorting the value head while
    allowing bust/FL heads to learn from diverse (including bad) states.
    """
    model.train()
    total_loss = 0
    n_batches = 0
    bce = nn.BCELoss()

    # Shuffle indices in-place
    perm = torch.randperm(len(indices), device='cpu')
    shuffled = indices[perm]

    for start in range(0, len(shuffled), batch_size):
        idx = shuffled[start:start + batch_size]
        obs = obs_gpu[idx].float()  # float16 → float32 per batch
        target = targets_gpu[idx]

        if use_turn:
            output = model(obs, turns_gpu[idx])
        else:
            output = model(obs)
        pred = output['value'].squeeze(-1)

        # Value loss: on-policy only if requested
        if value_onpolicy_only and onpolicy_gpu is not None:
            on_mask = onpolicy_gpu[idx]
            if on_mask.any():
                loss = criterion(pred[on_mask], target[on_mask])
            else:
                loss = torch.tensor(0.0, device=pred.device)
        else:
            loss = criterion(pred, target)

        # Bust/FL loss: always on ALL samples
        if bust_weight > 0:
            loss = loss + bust_weight * bce(output['bust_prob'].squeeze(-1), busted_gpu[idx])

        if fl_weight_loss > 0:
            loss = loss + fl_weight_loss * bce(output['fl_prob'].squeeze(-1), fl_gpu[idx])

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, score_mean, score_std):
    """Evaluate on validation set, return loss and correlation."""
    model.eval()
    total_loss = 0
    n_batches = 0
    all_preds = []
    all_targets = []

    for batch in loader:
        obs = batch[0].to(device)
        target = batch[1].to(device)

        output = model(obs)
        pred = output['value'].squeeze(-1)
        loss = criterion(pred, target)

        total_loss += loss.item()
        n_batches += 1

        all_preds.append(pred.cpu() * score_std + score_mean)
        all_targets.append(target.cpu() * score_std + score_mean)

    avg_loss = total_loss / max(n_batches, 1)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    if len(preds) > 1:
        p_mean = preds.mean()
        t_mean = targets.mean()
        cov = ((preds - p_mean) * (targets - t_mean)).mean()
        corr = (cov / (preds.std() * targets.std() + 1e-8)).item()
    else:
        corr = 0.0

    # Per-turn correlation
    all_turns = torch.cat([b[2] for b in loader_to_list(loader)])
    turn_corrs = {}
    for t in range(5):
        mask = all_turns == t
        if mask.sum() > 5:
            tp = preds[mask]
            tt = targets[mask]
            tc = ((tp - tp.mean()) * (tt - tt.mean())).mean()
            turn_corrs[t] = (tc / (tp.std() * tt.std() + 1e-8)).item()

    mae = (preds - targets).abs().mean().item()
    return avg_loss, corr, mae, turn_corrs


def loader_to_list(loader):
    """Helper to iterate loader and return all data."""
    result = []
    for batch in loader:
        result.append(batch)
    return result


@torch.no_grad()
def eval_epoch_simple(model, loader, criterion, device, score_mean, score_std,
                      use_turn=False):
    """Evaluation with per-turn bust/FL breakdown."""
    model.eval()
    total_loss = 0
    n_batches = 0
    all_preds = []
    all_targets = []
    all_bust_preds = []
    all_bust_labels = []
    all_fl_preds = []
    all_fl_labels = []
    all_turns = []

    for batch in loader:
        obs = batch[0].to(device)
        target = batch[1].to(device)
        turns = batch[2]

        if use_turn:
            output = model(obs, turns.to(device))
        else:
            output = model(obs)
        pred = output['value'].squeeze(-1)
        loss = criterion(pred, target)

        total_loss += loss.item()
        n_batches += 1
        all_preds.append(pred.cpu() * score_std + score_mean)
        all_targets.append(target.cpu() * score_std + score_mean)

        # Collect bust/FL predictions
        all_bust_preds.append(output['bust_prob'].squeeze(-1).cpu())
        all_bust_labels.append(batch[3])
        all_fl_preds.append(output['fl_prob'].squeeze(-1).cpu())
        all_fl_labels.append(batch[4])
        all_turns.append(turns)

    avg_loss = total_loss / max(n_batches, 1)
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    corr = 0.0
    if len(preds) > 1:
        p_mean = preds.mean()
        t_mean = targets.mean()
        cov = ((preds - p_mean) * (targets - t_mean)).mean()
        corr = (cov / (preds.std() * targets.std() + 1e-8)).item()

    mae = (preds - targets).abs().mean().item()

    # Bust/FL accuracy (overall)
    bust_preds = torch.cat(all_bust_preds)
    bust_labels = torch.cat(all_bust_labels)
    fl_preds = torch.cat(all_fl_preds)
    fl_labels = torch.cat(all_fl_labels)
    turns_all = torch.cat(all_turns)

    bust_acc = ((bust_preds > 0.5).float() == bust_labels).float().mean().item()
    fl_acc = ((fl_preds > 0.5).float() == fl_labels).float().mean().item()

    # Per-turn bust/FL metrics
    turn_metrics = {}
    for t in range(5):
        mask = turns_all == t
        n_t = mask.sum().item()
        if n_t < 10:
            continue
        bp = bust_preds[mask]
        bl = bust_labels[mask]
        fp = fl_preds[mask]
        fl = fl_labels[mask]
        b_acc = ((bp > 0.5).float() == bl).float().mean().item()
        f_acc = ((fp > 0.5).float() == fl).float().mean().item()
        # Bust recall: of actual busts, how many detected?
        n_bust = bl.sum().item()
        b_recall = ((bp > 0.5).float() * bl).sum().item() / max(n_bust, 1)
        # FL recall
        n_fl = fl.sum().item()
        f_recall = ((fp > 0.5).float() * fl).sum().item() / max(n_fl, 1)
        turn_metrics[t] = {
            'n': n_t, 'bust_acc': b_acc, 'bust_recall': b_recall,
            'fl_acc': f_acc, 'fl_recall': f_recall,
            'bust_rate': bl.mean().item(), 'fl_rate': fl.mean().item(),
        }

    return avg_loss, corr, mae, bust_acc, fl_acc, turn_metrics


@torch.no_grad()
def eval_epoch_gpu(obs_gpu, targets_gpu, turns_gpu, busted_gpu, fl_gpu,
                   indices, model, criterion, batch_size, score_mean, score_std,
                   use_turn=False):
    """GPU-preloaded evaluation with per-turn bust/FL breakdown."""
    model.eval()
    total_loss = 0
    n_batches = 0
    all_preds = []
    all_bust_preds = []
    all_fl_preds = []

    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        obs = obs_gpu[idx].float()  # float16 → float32 per batch
        target = targets_gpu[idx]

        if use_turn:
            output = model(obs, turns_gpu[idx])
        else:
            output = model(obs)
        pred = output['value'].squeeze(-1)
        loss = criterion(pred, target)

        total_loss += loss.item()
        n_batches += 1
        all_preds.append(pred.cpu() * score_std + score_mean)
        all_bust_preds.append(output['bust_prob'].squeeze(-1).cpu())
        all_fl_preds.append(output['fl_prob'].squeeze(-1).cpu())

    avg_loss = total_loss / max(n_batches, 1)
    preds = torch.cat(all_preds)
    targets_cpu = (targets_gpu[indices].cpu() * score_std + score_mean)
    bust_labels = busted_gpu[indices].cpu()
    fl_labels = fl_gpu[indices].cpu()
    turns_cpu = turns_gpu[indices].cpu()
    bust_preds = torch.cat(all_bust_preds)
    fl_preds = torch.cat(all_fl_preds)

    corr = 0.0
    if len(preds) > 1:
        p_mean = preds.mean()
        t_mean = targets_cpu.mean()
        cov = ((preds - p_mean) * (targets_cpu - t_mean)).mean()
        corr = (cov / (preds.std() * targets_cpu.std() + 1e-8)).item()

    mae = (preds - targets_cpu).abs().mean().item()
    bust_acc = ((bust_preds > 0.5).float() == bust_labels).float().mean().item()
    fl_acc = ((fl_preds > 0.5).float() == fl_labels).float().mean().item()

    turn_metrics = {}
    for t in range(5):
        mask = turns_cpu == t
        n_t = mask.sum().item()
        if n_t < 10:
            continue
        bp = bust_preds[mask]
        bl = bust_labels[mask]
        fp = fl_preds[mask]
        fl = fl_labels[mask]
        b_acc = ((bp > 0.5).float() == bl).float().mean().item()
        f_acc = ((fp > 0.5).float() == fl).float().mean().item()
        n_bust = bl.sum().item()
        b_recall = ((bp > 0.5).float() * bl).sum().item() / max(n_bust, 1)
        n_fl = fl.sum().item()
        f_recall = ((fp > 0.5).float() * fl).sum().item() / max(n_fl, 1)
        turn_metrics[t] = {
            'n': n_t, 'bust_acc': b_acc, 'bust_recall': b_recall,
            'fl_acc': f_acc, 'fl_recall': f_recall,
            'bust_rate': bl.mean().item(), 'fl_rate': fl.mean().item(),
        }

    return avg_loss, corr, mae, bust_acc, fl_acc, turn_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Value Network")
    parser.add_argument('--data', required=True, help='NPZ data file')
    parser.add_argument('--data-extra', nargs='*', default=[],
                        help='Additional NPZ files (e.g., off-policy data)')
    parser.add_argument('--max-samples', type=int, default=0,
                        help='Max samples per extra file (0=all, reduces RAM usage)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')
    parser.add_argument('--fl-weight', type=float, default=5.0,
                        help='Sample weight for FL hands (score > fl-threshold)')
    parser.add_argument('--fl-threshold', type=float, default=15.0,
                        help='Score threshold to identify FL hands')
    parser.add_argument('--bust-weight', type=float, default=0.0,
                        help='Loss weight for bust_prob BCE (0=disabled)')
    parser.add_argument('--fl-weight-loss', type=float, default=0.0,
                        help='Loss weight for fl_prob BCE (0=disabled)')
    parser.add_argument('--pretrained', default=None,
                        help='Path to pretrained model to initialize from')
    parser.add_argument('--save', default='ai/models/value_v1',
                        help='Directory to save model')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--v2', action='store_true',
                        help='Use V2 architecture (ResBlock + LayerNorm)')
    parser.add_argument('--v3', action='store_true',
                        help='Use V3 architecture (Turn-Conditioned Heads)')
    parser.add_argument('--value-onpolicy-only', action='store_true',
                        help='Compute value loss only on on-policy samples (--data). '
                             'Bust/FL loss still uses all data. '
                             'Prevents off-policy data from distorting value head.')
    args = parser.parse_args()

    use_turn = args.v3  # V3 passes turn to model

    print("=" * 60)
    print("  Value Network Training")
    print("=" * 60)
    print(f"  Data: {args.data}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR: {args.lr}")
    print(f"  FL weight: {args.fl_weight}x (threshold={args.fl_threshold})")
    multitask = args.bust_weight > 0 or args.fl_weight_loss > 0
    if multitask:
        print(f"  Multi-task: bust_weight={args.bust_weight}, fl_weight_loss={args.fl_weight_loss}")
    if args.value_onpolicy_only:
        print(f"  Value head: ON-POLICY ONLY (bust/FL heads use all data)")
    if args.pretrained:
        print(f"  Pretrained: {args.pretrained}")
    print(f"  Device: {args.device}")
    print(f"  Save: {args.save}")
    print()

    # Load data
    data_paths = [args.data] + args.data_extra
    dataset = ValueDataset(data_paths, max_samples_per_file=args.max_samples)

    # Split train/val
    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val

    # GPU preload mode: move all tensors to GPU, use manual batching
    use_gpu_preload = args.device == 'cuda'

    if use_gpu_preload:
        print(f"  GPU preload mode: moving {len(dataset):,} samples to GPU...")
        obs_gpu = dataset.obs.half().to(args.device)       # float16 on GPU (saves VRAM)
        targets_gpu = dataset.targets.to(args.device)
        turns_gpu = dataset.turns.to(args.device)
        busted_gpu = dataset.busted.to(args.device)
        fl_gpu = dataset.fl_entry.to(args.device)
        onpolicy_gpu = dataset.is_onpolicy.to(args.device) if args.value_onpolicy_only else None

        # Split indices
        all_idx = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))
        train_idx = all_idx[:n_train]
        val_idx = all_idx[n_train:]

        train_loader = None
        val_loader = None
        print(f"  Train: {n_train}, Val: {n_val} (GPU preloaded)")
    else:
        train_set, val_set = random_split(dataset, [n_train, n_val],
                                           generator=torch.Generator().manual_seed(42))
        if args.fl_weight > 1.0:
            all_weights = dataset.get_sample_weights(args.fl_threshold, args.fl_weight)
            train_weights = all_weights[train_set.indices]
            sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)
            train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler,
                                      num_workers=0, pin_memory=True)
        else:
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                      num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                num_workers=0, pin_memory=True)
        print(f"  Train: {n_train}, Val: {n_val}")

    # Model — detect input dim from data
    obs_dim = dataset.obs.shape[1]
    if args.v3:
        model = ValueNetworkV3(input_dim=obs_dim).to(args.device)
        print(f"  Using V3 architecture (Turn-Conditioned Heads, input_dim={obs_dim})")
    elif args.v2:
        model = ValueNetworkV2().to(args.device)
        print(f"  Using V2 architecture (ResBlock)")
    else:
        model = ValueNetwork().to(args.device)

    # Load pretrained weights if specified
    if args.pretrained and Path(args.pretrained).exists():
        ck = torch.load(args.pretrained, map_location=args.device, weights_only=False)
        sd = ck.get('model_state_dict', ck)
        if args.v3:
            # V3: load compatible weights from V1 checkpoint
            n_loaded = model.load_v1_weights(sd)
            print(f"  Loaded {n_loaded} compatible weights from {args.pretrained} (V1→V3)")
        else:
            model.load_state_dict(sd)
            print(f"  Loaded pretrained weights from {args.pretrained}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    # Training loop
    best_val_loss = float('inf')
    best_corr = 0.0
    patience_counter = 0
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        if use_gpu_preload:
            train_loss = train_epoch_gpu(
                obs_gpu, targets_gpu, turns_gpu, busted_gpu, fl_gpu,
                train_idx, model, optimizer, criterion, args.batch_size,
                bust_weight=args.bust_weight, fl_weight_loss=args.fl_weight_loss,
                use_turn=use_turn,
                onpolicy_gpu=onpolicy_gpu, value_onpolicy_only=args.value_onpolicy_only)
            val_loss, val_corr, val_mae, bust_acc, fl_acc, turn_metrics = eval_epoch_gpu(
                obs_gpu, targets_gpu, turns_gpu, busted_gpu, fl_gpu,
                val_idx, model, criterion, args.batch_size,
                dataset.score_mean, dataset.score_std, use_turn=use_turn)
        else:
            train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device,
                                     bust_weight=args.bust_weight,
                                     fl_weight_loss=args.fl_weight_loss,
                                     use_turn=use_turn)
            val_loss, val_corr, val_mae, bust_acc, fl_acc, turn_metrics = eval_epoch_simple(
                model, val_loader, criterion, args.device,
                dataset.score_mean, dataset.score_std,
                use_turn=use_turn)
        scheduler.step()

        # Check improvement
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_corr = val_corr
            patience_counter = 0

            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_corr': val_corr,
                'val_mae': val_mae,
                'score_mean': dataset.score_mean,
                'score_std': dataset.score_std,
                'norm_stats': {'mean': dataset.score_mean, 'std': dataset.score_std},
                'train_samples': n_train,
                'arch': 'v3' if args.v3 else ('v2' if args.v2 else 'v1'),
            }, save_dir / 'value_best.pt')
        else:
            patience_counter += 1

        # Logging
        marker = " *" if improved else ""
        if epoch % 10 == 0 or epoch <= 5 or improved:
            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]
            extra = ""
            if multitask and dataset.has_labels:
                extra = f" bust_acc={bust_acc:.2f} fl_acc={fl_acc:.2f}"
            print(f"  [{epoch:3d}/{args.epochs}] "
                  f"train={train_loss:.4f} val={val_loss:.4f} "
                  f"corr={val_corr:.3f} mae={val_mae:.2f} "
                  f"lr={lr_now:.1e} ({elapsed:.0f}s){extra}{marker}")

            # Per-turn breakdown every 10 epochs or on improvement
            if (epoch % 10 == 0 or epoch <= 3) and turn_metrics:
                for t in sorted(turn_metrics):
                    m = turn_metrics[t]
                    print(f"    T{t}: n={m['n']:>6,}  "
                          f"bust_acc={m['bust_acc']:.2f} recall={m['bust_recall']:.2f} "
                          f"(rate={m['bust_rate']:.1%})  "
                          f"fl_acc={m['fl_acc']:.2f} recall={m['fl_recall']:.2f} "
                          f"(rate={m['fl_rate']:.1%})")

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch} (patience={args.patience})")
            break

    elapsed = time.time() - t0

    # Final per-turn evaluation
    if use_gpu_preload:
        _, _, _, _, _, final_metrics = eval_epoch_gpu(
            obs_gpu, targets_gpu, turns_gpu, busted_gpu, fl_gpu,
            val_idx, model, criterion, args.batch_size,
            dataset.score_mean, dataset.score_std, use_turn=use_turn)
    else:
        _, _, _, _, _, final_metrics = eval_epoch_simple(
            model, val_loader, criterion, args.device,
            dataset.score_mean, dataset.score_std, use_turn=use_turn)

    print(f"\n{'=' * 60}")
    print(f"  Training Complete")
    print(f"{'=' * 60}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best correlation: {best_corr:.3f}")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"  Model saved to: {save_dir / 'value_best.pt'}")

    if final_metrics:
        print(f"\n  Per-turn bust/FL evaluation (final model):")
        print(f"  {'Turn':>4}  {'N':>7}  {'Bust%':>6}  {'BustAcc':>7}  {'BustRec':>7}  "
              f"{'FL%':>5}  {'FLAcc':>5}  {'FLRec':>5}")
        for t in sorted(final_metrics):
            m = final_metrics[t]
            print(f"  T{t:>3}  {m['n']:>7,}  {m['bust_rate']:>5.1%}  {m['bust_acc']:>7.2f}  "
                  f"{m['bust_recall']:>7.2f}  {m['fl_rate']:>5.1%}  {m['fl_acc']:>5.2f}  "
                  f"{m['fl_recall']:>5.2f}")

    # Save normalization stats separately for inference
    import json
    stats = {
        'score_mean': dataset.score_mean,
        'score_std': dataset.score_std,
        'n_samples': len(dataset),
        'arch': 'v3' if args.v3 else ('v2' if args.v2 else 'v1'),
    }
    with open(save_dir / 'norm_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Norm stats saved to: {save_dir / 'norm_stats.json'}")


if __name__ == '__main__':
    main()
