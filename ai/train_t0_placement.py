#!/usr/bin/env python3
"""
T0 Placement Network Training (v4 - Config F).

Improvements over v3:
- Hand-level features (pair/flush/straight detection)
- Suit permutation augmentation (24x)
- Moderate regularization (dropout=0.2, wd=0.02)
- Proper hand-level train/val split
"""

import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from itertools import permutations
from collections import Counter


CARD_DIM_BASE = 18  # 13 ranks + 4 suits + 1 joker flag
HAND_FEAT_DIM = 6   # pair, trips, flush_draw, straight_conn, high_card, royalty
CARD_DIM = CARD_DIM_BASE + HAND_FEAT_DIM  # 24 total
NUM_ROWS = 3   # Top=0, Mid=1, Bot=2
MAX_CARDS = 5
SUITS = ['spades', 'hearts', 'diamonds', 'clubs']


def encode_card(card: dict) -> np.ndarray:
    """Encode a single card dict into base features."""
    features = np.zeros(CARD_DIM_BASE, dtype=np.float32)
    rank = card.get('rank', '')
    suit = card.get('suit', '')

    if rank == 'Joker' or suit == 'joker':
        features[17] = 1.0  # Joker flag
    else:
        rank_map = {'2':0,'3':1,'4':2,'5':3,'6':4,'7':5,
                   '8':6,'9':7,'T':8,'J':9,'Q':10,'K':11,'A':12}
        r = rank_map.get(rank, 0)
        features[r] = 1.0
        suit_map = {'spades':0,'hearts':1,'diamonds':2,'clubs':3}
        s = suit_map.get(suit, 0)
        features[13 + s] = 1.0

    return features


def compute_hand_features(hand):
    """Compute hand-level features for 5 cards. Returns (5, HAND_FEAT_DIM)."""
    rank_map = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,
                '8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}
    ranks = [rank_map.get(c.get('rank',''), 0) for c in hand]
    suits = [c.get('suit','') for c in hand]
    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)

    extras = np.zeros((5, HAND_FEAT_DIM), dtype=np.float32)
    for i, (r, s) in enumerate(zip(ranks, suits)):
        extras[i, 0] = 1.0 if rank_counts[r] >= 2 else 0.0  # pair member
        extras[i, 1] = 1.0 if rank_counts[r] >= 3 else 0.0  # trips member
        extras[i, 2] = 1.0 if suit_counts[s] >= 3 else 0.0  # flush draw
        extras[i, 3] = 1.0 if any(abs(r - r2) == 1 for j, r2 in enumerate(ranks) if j != i) else 0.0
        extras[i, 4] = 1.0 if r >= 10 else 0.0  # high card
        extras[i, 5] = 1.0 if r >= 12 and rank_counts[r] >= 2 else 0.0  # royalty potential
    return extras


def card_key(card):
    return f"{card['rank']}_{card['suit']}"


def permute_card(card, suit_map):
    new = dict(card)
    s = card.get('suit', '')
    if s in suit_map:
        new['suit'] = suit_map[s]
    return new


def augment_samples(samples):
    """Apply 24x suit permutation augmentation."""
    all_perms = list(permutations(SUITS))
    augmented = []
    for sample in samples:
        seen = set()
        for perm in all_perms:
            suit_map = {SUITS[i]: perm[i] for i in range(4)}
            new_hand = [permute_card(c, suit_map) for c in sample['hand']]
            hand_k = tuple((c['rank'], c['suit']) for c in new_hand)
            if hand_k in seen:
                continue
            seen.add(hand_k)
            sol = sample.get('solution', {})
            new_sample = {
                'hand': new_hand,
                'solution': {
                    'top': [permute_card(c, suit_map) for c in sol.get('top', [])],
                    'mid': [permute_card(c, suit_map) for c in sol.get('mid', [])],
                    'bot': [permute_card(c, suit_map) for c in sol.get('bot', [])],
                },
                'placements': [{
                    'top': [permute_card(c, suit_map) for c in p.get('top', [])],
                    'mid': [permute_card(c, suit_map) for c in p.get('mid', [])],
                    'bot': [permute_card(c, suit_map) for c in p.get('bot', [])],
                    'ev': p.get('ev', 0),
                } for p in sample.get('placements', [])],
                'best_ev': sample.get('best_ev', 0),
            }
            augmented.append(new_sample)
    return augmented


class T0PlacementDataset(Dataset):
    """Dataset for T0 placement learning with hand features."""

    def __init__(self, samples, top_k=10):
        self.samples = []
        for s in samples:
            p = self._process(s, top_k)
            if p:
                self.samples.append(p)
        print(f"  Dataset: {len(self.samples)} samples")

    def _process(self, data, top_k):
        hand = data.get('hand', [])
        if len(hand) != 5:
            return None

        base_features = np.stack([encode_card(c) for c in hand])
        hand_features = compute_hand_features(hand)
        features = np.concatenate([base_features, hand_features], axis=1)

        placements = data.get('placements', [])
        if not placements:
            return None

        placements.sort(key=lambda x: x.get('ev', -999), reverse=True)
        top_p = placements[:top_k]
        evs = np.array([p['ev'] for p in top_p], dtype=np.float32)
        ev_w = np.exp(evs - evs.max())
        ev_w /= ev_w.sum()

        soft_labels = np.zeros((5, NUM_ROWS), dtype=np.float32)
        hand_keys = [card_key(c) for c in hand]
        for p, w in zip(top_p, ev_w):
            cr = {}
            for rn, ri in [('top',0),('mid',1),('bot',2)]:
                for card in p.get(rn,[]):
                    cr[card_key(card)] = ri
            for i, key in enumerate(hand_keys):
                if key in cr:
                    soft_labels[i, cr[key]] += w
        rs = soft_labels.sum(axis=1, keepdims=True)
        rs = np.where(rs == 0, 1.0, rs)
        soft_labels /= rs

        best = placements[0]
        hard_labels = np.zeros(5, dtype=np.int64)
        bcr = {}
        for rn, ri in [('top',0),('mid',1),('bot',2)]:
            for card in best.get(rn,[]):
                bcr[card_key(card)] = ri
        for i, key in enumerate(hand_keys):
            hard_labels[i] = bcr.get(key, 2)

        return {
            'features': features,
            'soft_labels': soft_labels,
            'hard_labels': hard_labels,
            'best_ev': np.float32(data.get('best_ev', 0.0)),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        f = s['features'].copy()
        sl = s['soft_labels'].copy()
        hl = s['hard_labels'].copy()
        perm = np.random.permutation(5)
        return {
            'features': torch.from_numpy(f[perm]),
            'soft_labels': torch.from_numpy(sl[perm]),
            'hard_labels': torch.from_numpy(hl[perm]),
            'best_ev': torch.tensor(s['best_ev']),
        }


class T0PlacementNet(nn.Module):
    """Transformer for T0 card placement (Config F: hand features + moderate reg)."""

    def __init__(self, d_model=128, nhead=4, num_layers=4, dim_ff=256, dropout=0.2):
        super().__init__()
        self.card_embed = nn.Sequential(
            nn.Linear(CARD_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, MAX_CARDS, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.row_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, NUM_ROWS),
        )
        self.ev_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, cards):
        x = self.card_embed(cards) + self.pos_embed[:, :cards.size(1)]
        x = self.encoder(x)
        row_logits = self.row_head(x)
        ev_pred = self.ev_head(x.mean(dim=1))
        return row_logits, ev_pred


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = total_card = total_hand = 0
    for batch in loader:
        features = batch['features'].to(device)
        soft_labels = batch['soft_labels'].to(device)
        hard_labels = batch['hard_labels'].to(device)
        best_ev = batch['best_ev'].to(device)
        optimizer.zero_grad()
        logits, ev_pred = model(features)
        log_probs = F.log_softmax(logits, dim=-1)
        soft_loss = -(soft_labels * log_probs).sum(dim=-1).mean()
        hard_loss = F.cross_entropy(logits.reshape(-1, NUM_ROWS), hard_labels.reshape(-1))
        ev_loss = F.mse_loss(ev_pred.squeeze(-1), best_ev)
        loss = 0.7 * soft_loss + 0.3 * hard_loss + 0.1 * ev_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        card_c = (pred == hard_labels).float()
        total_card += card_c.mean().item()
        total_hand += card_c.prod(dim=1).mean().item()
    n = len(loader)
    return total_loss/n, total_card/n, total_hand/n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = total_card = total_hand = 0
    for batch in loader:
        features = batch['features'].to(device)
        soft_labels = batch['soft_labels'].to(device)
        hard_labels = batch['hard_labels'].to(device)
        best_ev = batch['best_ev'].to(device)
        logits, ev_pred = model(features)
        log_probs = F.log_softmax(logits, dim=-1)
        soft_loss = -(soft_labels * log_probs).sum(dim=-1).mean()
        hard_loss = F.cross_entropy(logits.reshape(-1, NUM_ROWS), hard_labels.reshape(-1))
        ev_loss = F.mse_loss(ev_pred.squeeze(-1), best_ev)
        loss = 0.7 * soft_loss + 0.3 * hard_loss + 0.1 * ev_loss
        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        card_c = (pred == hard_labels).float()
        total_card += card_c.mean().item()
        total_hand += card_c.prod(dim=1).mean().item()
    n = len(loader)
    return total_loss/n, total_card/n, total_hand/n


def main():
    parser = argparse.ArgumentParser(description="Train T0 Placement Network v4 (Config F)")
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--weight-decay', type=float, default=0.02)
    parser.add_argument('--output', type=str, default='ai/models/t0_placement_net.pt')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--top-k', type=int, default=10)
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Config F: d_model={args.d_model}, layers={args.num_layers}, "
          f"dropout={args.dropout}, wd={args.weight_decay}, hand_features=True")

    # Load all unique hands
    all_samples = []
    with open(args.data, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_samples.append(json.loads(line.strip()))
    print(f"Total unique hands: {len(all_samples)}")

    # Hand-level split
    random.seed(42)
    random.shuffle(all_samples)
    split = int(0.9 * len(all_samples))
    train_raw = all_samples[:split]
    val_raw = all_samples[split:]
    print(f"Train hands: {len(train_raw)}, Val hands: {len(val_raw)}")

    # Suit augmentation
    train_aug = augment_samples(train_raw)
    val_aug = augment_samples(val_raw)
    print(f"Train augmented: {len(train_aug)}, Val augmented: {len(val_aug)}")

    train_ds = T0PlacementDataset(train_aug, top_k=args.top_k)
    val_ds = T0PlacementDataset(val_aug, top_k=args.top_k)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = T0PlacementNet(
        d_model=args.d_model, nhead=4, num_layers=args.num_layers,
        dim_ff=args.d_model * 2, dropout=args.dropout,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    best_val_hand = 0
    patience = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_card, train_hand = train_epoch(model, train_loader, optimizer, args.device)
        val_loss, val_card, val_hand = evaluate(model, val_loader, args.device)
        scheduler.step()

        if epoch % 5 == 0 or epoch <= 5:
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Train L={train_loss:.4f} Card={train_card:.3f} Hand={train_hand:.3f} | "
                  f"Val L={val_loss:.4f} Card={val_card:.3f} Hand={val_hand:.3f}")

        if val_hand > best_val_hand:
            best_val_hand = val_hand
            patience = 0
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_card_acc': val_card,
                'val_hand_acc': val_hand,
                'config': {
                    'd_model': args.d_model,
                    'num_layers': args.num_layers,
                    'dropout': args.dropout,
                    'input_dim': CARD_DIM,
                    'hand_features': True,
                    'n_params': n_params,
                    'version': 'v4_config_f',
                },
            }, args.output)
            print(f"  >>> Saved best (hand_acc={val_hand:.4f})")
        else:
            patience += 1
            if patience >= 30:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nTraining complete. Best val hand accuracy: {best_val_hand:.4f}")
    print(f"Model saved to: {args.output}")


if __name__ == '__main__':
    main()
