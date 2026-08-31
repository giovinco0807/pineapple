"""
T3 Oracle Training v2 — Improved precision

Key improvements over v1:
  1. Full 522-dim state input (486 cards + 6 meta + 30 game-aware)
  2. Suit isomorphism data augmentation (24x effective data)
  3. Warmup + Cosine Annealing LR schedule
  4. Temperature scheduling (high→low over training)
  5. 27-slot action space (3 discards × 9 placements) with true per-action EVs

Data source: D:/ofc_data/t3_train_27slot/
  states:      (996000, 522) float16
  action_evs:  (996000, 27)  float16
  valid_masks: (996000, 27)  bool

Usage:
  python ai/training/train_t3_oracle_v2.py --epochs 500
"""
import sys
import argparse
import json
import time
import itertools
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# Model (identical architecture, but state_dim=520)
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


class T3PolicyValueNet(nn.Module):
    """
    PolicyValueNet with full 520-dim input.

    Input:  520-dim state (486 card matrix + 4 meta + 30 game-aware)
    Policy: 27 logits (action ranking)
    Value:  1 scalar (best EV prediction)
    """

    def __init__(self, state_dim: int = 520, n_actions: int = 27,
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
# Suit Isomorphism Augmentation
# ─────────────────────────────────────────────────────────────────────────────

# Pre-build all 24 suit permutation index maps for the 54-card matrix
# Card layout: 4 suits × 13 ranks + 2 jokers (X1, X2)
# Original order: h(0-12), d(13-25), c(26-38), s(39-51), X1(52), X2(53)
SUIT_ORDER = 'hdcs'
ALL_PERMS = list(itertools.permutations(range(4)))  # 24 permutations

def _build_perm_indices():
    """Build card index remapping for each of the 24 suit permutations."""
    perm_maps = []
    for perm in ALL_PERMS:
        # perm[i] = new suit index for original suit i
        mapping = np.zeros(54, dtype=np.int64)
        for orig_suit in range(4):
            new_suit = perm[orig_suit]
            for rank in range(13):
                orig_idx = orig_suit * 13 + rank
                new_idx  = new_suit * 13 + rank
                mapping[orig_idx] = new_idx
        mapping[52] = 52  # joker unchanged
        mapping[53] = 53  # joker unchanged
        perm_maps.append(mapping)
    return perm_maps

PERM_MAPS = _build_perm_indices()


def augment_batch_suits(states_batch, perm_idx):
    """
    Apply suit permutation to a batch of 520-dim states.
    
    Card matrix (dims 0-485): 54 cards × 9 locations → remap card indices
    Meta (dims 486-491): turn, is_btn, is_fl, opp_is_fl, chips_self, chips_opp → unchanged
    Game-aware (dims 492-521): 
      - Row slots (492-497): unchanged (count-based)
      - FL features (498-503): unchanged (rank-based, suit-independent)
      - Opp FL features (504-507): unchanged
      - Hand ranks (508-513): unchanged (rank-based)
      - Bust risk (514-515): unchanged
      - Draw features (516-521): Flush draws are suit-dependent!
        → We zero them out since they become invalid after permutation.
        The model should learn to reconstruct flush info from the card matrix.
    """
    perm_map = PERM_MAPS[perm_idx]
    batch_size = states_batch.shape[0]
    new_states = states_batch.clone()
    
    # Reshape card matrix portion: (batch, 54, 9) 
    card_part = states_batch[:, :486].reshape(batch_size, 54, 9)
    # Remap: new_card_part[:, new_idx, :] = old_card_part[:, orig_idx, :]
    perm_tensor = torch.from_numpy(perm_map).to(states_batch.device)
    new_card = torch.zeros_like(card_part)
    new_card[:, perm_tensor, :] = card_part
    new_states[:, :486] = new_card.reshape(batch_size, 486)
    
    # Zero out flush draw features (they depend on suit groupings)
    # 522-dim layout: 486 cards + 6 meta + 6 row_slots + 6 fl_self + 4 fl_opp 
    #                 + 6 hand_ranks + 2 bust_risk + 6 draws
    # Draws (dims 516-521): flush_mid, flush_bot, str_mid, str_bot, pair_mid, pair_bot
    # Flush draws at 516, 517 change after suit permutation → zero them out.
    # The card matrix already encodes complete suit information.
    new_states[:, 516] = 0.0  # flush_draw_mid  
    new_states[:, 517] = 0.0  # flush_draw_bot
    
    return new_states


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Loader (from NPI files)
# ─────────────────────────────────────────────────────────────────────────────

class T3DatasetNPI:
    """Load T3 training data from D:/ofc_data/t3_train/ NPI files."""

    def __init__(self, data_dir, n_actions=27, device='cpu', verbose=True):
        data_dir = Path(data_dir)
        
        if verbose:
            print(f"  Loading from {data_dir}...")
        
        states_path = data_dir / 'states.npy'
        if states_path.exists():
            states_raw = np.load(states_path, mmap_mode='r')
            evs_raw    = np.load(data_dir / 'action_evs.npy', mmap_mode='r')
            masks_raw  = np.load(data_dir / 'valid_masks.npy', mmap_mode='r')
        else:
            files = sorted(data_dir.glob('*.npz'))
            if not files:
                raise FileNotFoundError(f"No states.npy or NPZ chunks found in {data_dir}")

            state_parts, ev_parts, mask_parts = [], [], []
            for f in files:
                npz = np.load(f)
                state_parts.append(npz['states'])
                ev_parts.append(npz['action_evs'])
                if 'valid_masks' in npz:
                    mask_parts.append(npz['valid_masks'])
                else:
                    mask_parts.append(npz['action_masks'])

            states_raw = np.concatenate(state_parts, axis=0)
            evs_raw    = np.concatenate(ev_parts, axis=0)
            masks_raw  = np.concatenate(mask_parts, axis=0)
        
        N = states_raw.shape[0]
        state_dim = states_raw.shape[1]  # 520
        
        if verbose:
            print(f"  Raw: {N:,} samples, state_dim={state_dim}")
        
        # Load full states (520 dim)
        states = np.array(states_raw[:], dtype=np.float32)
        
        # Keep the fixed semantic action slots used by regular turns.
        evs   = np.array(evs_raw[:, :n_actions], dtype=np.float32)
        masks = np.array(masks_raw[:, :n_actions])
        
        # Compute best EVs from the valid actions
        evs_for_best = evs.copy()
        evs_for_best[~masks] = -1e9
        bests = evs_for_best.max(axis=1).astype(np.float32)
        
        self.states = torch.from_numpy(states).to(device)
        self.evs    = torch.from_numpy(evs).to(device)
        self.masks  = torch.from_numpy(masks).to(device)
        self.bests  = torch.from_numpy(bests).to(device)

        if verbose:
            valid_evs = evs[evs > -1e8]
            print(f"  Loaded {N:,} samples (state_dim={state_dim}, n_actions={n_actions})")
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

def train(args):
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"  T3 Oracle Training v2 (Full 520-dim + Suit Augmentation)")
    print(f"{'='*60}")
    print(f"  Device: {device}")

    # ── Load data ──
    ds = T3DatasetNPI(args.data_dir, n_actions=27, device=device)
    N = len(ds)
    state_dim = ds.states.shape[1]

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
    model = T3PolicyValueNet(
        state_dim=state_dim,
        n_actions=ds.evs.shape[1],
        hidden=args.hidden,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
    ).to(device)

    if args.resume_model:
        ckpt = torch.load(args.resume_model, map_location=device)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        print(f"  Resumed model: {args.resume_model}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: T3PolicyValueNet ({n_params:,} params, input={state_dim})")
    print(f"  Suit augmentation: {'ON' if args.suit_augment else 'OFF'}")
    print(f"  Temperature schedule: {args.temp_start} → {args.temp_end}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Warmup + Cosine Annealing schedule
    warmup_epochs = min(10, args.epochs // 10)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # linear warmup
        else:
            # Cosine decay from 1.0 to 0.01
            progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
            return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_regret = float('inf')
    best_epoch = 0
    history = []

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        
        # Temperature schedule: linear decay
        progress = (epoch - 1) / max(1, args.epochs - 1)
        temperature = args.temp_start + (args.temp_end - args.temp_start) * progress

        # Shuffle train indices
        perm_e = train_idx[torch.randperm(len(train_idx))]
        train_policy_loss = 0.0
        train_value_loss  = 0.0
        n_batches = 0

        # Choose suit permutation for this epoch (if augmenting)
        if args.suit_augment:
            # Cycle through all 24 permutations; identity (perm 0) included
            suit_perm_idx = epoch % 24
        else:
            suit_perm_idx = 0  # identity

        for i in range(0, len(perm_e), args.batch_size):
            idx = perm_e[i:i + args.batch_size]
            states = ds.states[idx]
            evs    = ds.evs[idx]
            masks  = ds.masks[idx]
            bests  = ds.bests[idx]

            # Apply suit augmentation (skip identity perm for speed)
            if args.suit_augment and suit_perm_idx != 0:
                states = augment_batch_suits(states, suit_perm_idx)

            logits, value = model(states, masks)

            # Policy loss: ListNet ranking with scheduled temperature
            p_loss = listnet_loss(logits, evs, masks, temperature)
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
                    states = ds.states[idx]
                    evs    = ds.evs[idx]
                    masks  = ds.masks[idx]
                    bests  = ds.bests[idx]

                    logits, value = model(states, masks)
                    val_policy_loss += listnet_loss(logits, evs, masks, temperature).item()
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
                'temperature': temperature,
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
                    'state_dim':  state_dim,
                    'n_actions':  ds.evs.shape[1],
                    'hidden':     args.hidden,
                    'n_blocks':   args.n_blocks,
                }, save_dir / 't3_policyvalue_v2_best.pt')

            print(
                f"  [{epoch:3d}/{args.epochs}] "
                f"p_loss={record['train_policy_loss']:.4f} "
                f"v_loss={record['train_value_loss']:.4f} | "
                f"val_regret={val_regret:.3f} "
                f"top1={val_top1:.1%} "
                f"T={temperature:.2f} "
                f"lr={lr_now:.1e} "
                f"({elapsed:.0f}s){marker}"
            )

    # ── Final save ──
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'val_regret': best_val_regret,
        'state_dim': state_dim,
        'n_actions': ds.evs.shape[1],
        'hidden': args.hidden,
        'n_blocks': args.n_blocks,
    }, save_dir / 't3_policyvalue_v2_final.pt')

    with open(save_dir / 't3_training_history_v2.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training Complete")
    print(f"{'='*60}")
    print(f"  Best val regret: {best_val_regret:.3f} at epoch {best_epoch}")
    print(f"  Model saved: {save_dir / 't3_policyvalue_v2_best.pt'}")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train T3 Oracle v2 (Improved)')
    parser.add_argument('--data-dir',     default='ai/data/t3_dataset_bi',
                        help='Dir containing states.npy, action_evs.npy, valid_masks.npy')
    parser.add_argument('--save-dir',     default='ai/data/t3_oracle_bi',
                        help='Where to save model and history')
    parser.add_argument('--epochs',       type=int,   default=500)
    parser.add_argument('--batch-size',   type=int,   default=2048)
    parser.add_argument('--lr',           type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--hidden',       type=int,   default=1024)
    parser.add_argument('--n-blocks',     type=int,   default=4)
    parser.add_argument('--dropout',      type=float, default=0.1)
    parser.add_argument('--temp-start',   type=float, default=3.0,
                        help='ListNet temperature at start')
    parser.add_argument('--temp-end',     type=float, default=0.5,
                        help='ListNet temperature at end')
    parser.add_argument('--value-weight', type=float, default=0.3,
                        help='Weight for value head MSE loss')
    parser.add_argument('--resume-model', default='',
                        help='Checkpoint to initialize model weights from')
    parser.add_argument('--eval-every',   type=int,   default=5,
                        help='Evaluate every N epochs')
    parser.add_argument('--suit-augment', action='store_true', default=True,
                        help='Enable suit isomorphism augmentation')
    parser.add_argument('--no-suit-augment', dest='suit_augment', action='store_false')
    parser.add_argument('--device',       default='auto')
    args = parser.parse_args()

    train(args)
