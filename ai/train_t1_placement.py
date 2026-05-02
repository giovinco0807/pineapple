#!/usr/bin/env python3
"""
T1 Placement Network Training.
Takes 8 cards (5 board + 3 hand). Predicts 4 classes (Top, Mid, Bot, Discard) for each hand card.
"""

import argparse
import json
import random
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from itertools import permutations

CARD_DIM_BASE = 18  # 13 ranks + 4 suits + 1 joker flag
BOARD_ROW_DIM = 4   # 0: Hand, 1: Top, 2: Mid, 3: Bot
CARD_DIM = CARD_DIM_BASE + BOARD_ROW_DIM  # 22
NUM_CLASSES = 4     # Top=0, Mid=1, Bot=2, Discard=3
MAX_CARDS = 8       # 5 on board + 3 in hand
SUITS = ['s', 'h', 'd', 'c']

def encode_card_str(card_str: str, board_row: int) -> np.ndarray:
    """Encode a card string like '9s' or 'JK' into base features + board row."""
    features = np.zeros(CARD_DIM, dtype=np.float32)
    if card_str == 'JK':
        features[17] = 1.0
    else:
        rank = card_str[0]
        suit_char = card_str[1]
        rank_map = {'2':0,'3':1,'4':2,'5':3,'6':4,'7':5,
                   '8':6,'9':7,'T':8,'J':9,'Q':10,'K':11,'A':12}
        r = rank_map.get(rank, 0)
        features[r] = 1.0
        suit_map = {'s':0,'h':1,'d':2,'c':3}
        s = suit_map.get(suit_char, 0)
        features[13 + s] = 1.0
    features[18 + board_row] = 1.0
    return features

def permute_card_str(c_str, suit_map):
    if c_str == 'JK': return 'JK'
    return c_str[0] + suit_map[c_str[1]]

def parse_board(board_str):
    top_match = re.search(r'Top\[(.*?)\]', board_str)
    mid_match = re.search(r'Mid\[(.*?)\]', board_str)
    bot_match = re.search(r'Bot\[(.*?)\]', board_str)
    top = top_match.group(1).split() if top_match else []
    mid = mid_match.group(1).split() if mid_match else []
    bot = bot_match.group(1).split() if bot_match else []
    return top, mid, bot

def augment_samples(samples):
    """Apply 24x suit permutation augmentation."""
    all_perms = list(permutations(SUITS))
    augmented = []
    for sample in samples:
        seen = set()
        for perm in all_perms:
            suit_map = {SUITS[i]: perm[i] for i in range(4)}
            new_hand = [permute_card_str(c, suit_map) for c in sample['hand'].split()]
            hand_k = tuple(sorted(new_hand))
            if hand_k in seen:
                continue
            seen.add(hand_k)
            
            top, mid, bot = parse_board(sample['board'])
            new_top = " ".join([permute_card_str(c, suit_map) for c in top])
            new_mid = " ".join([permute_card_str(c, suit_map) for c in mid])
            new_bot = " ".join([permute_card_str(c, suit_map) for c in bot])
            new_board = f"Top[{new_top}] Mid[{new_mid}] Bot[{new_bot}]"
            
            new_placements = []
            for p in sample.get('placements', []):
                new_d = permute_card_str(p['d'], suit_map)
                new_p_parts = []
                if p['p']:
                    for part in p['p'].split(', '):
                        if part:
                            c, t = part.split('→')
                            new_p_parts.append(f"{permute_card_str(c, suit_map)}→{t}")
                new_placements.append({
                    'd': new_d,
                    'p': ", ".join(new_p_parts),
                    'ev': p.get('ev', 0)
                })
            
            augmented.append({
                'board': new_board,
                'hand': " ".join(new_hand),
                'placements': new_placements,
                'original_ev': sample.get('original_ev', 0)
            })
    return augmented


class T1PlacementDataset(Dataset):
    """Dataset for T1 placement learning."""

    def __init__(self, samples, top_k=10):
        self.samples = []
        for s in samples:
            p = self._process(s, top_k)
            if p:
                self.samples.append(p)
        print(f"  Dataset: {len(self.samples)} samples")

    def _process(self, data, top_k):
        top, mid, bot = parse_board(data['board'])
        hand = data['hand'].split()
        
        board_cards = []
        for c in top: board_cards.append((c, 1))
        for c in mid: board_cards.append((c, 2))
        for c in bot: board_cards.append((c, 3))
        
        hand_cards = [(c, 0) for c in hand]
        all_cards = board_cards + hand_cards
        if len(all_cards) != 8:
            # Usually exactly 5 on board and 3 in hand. If not, pad or skip?
            # T1 should always be 5 + 3 = 8
            pass
            
        features = np.stack([encode_card_str(c, r) for c, r in all_cards])
        
        placements = data.get('placements', [])
        if not placements:
            return None

        placements.sort(key=lambda x: x.get('ev', -999), reverse=True)
        top_p = placements[:top_k]
        evs = np.array([p['ev'] for p in top_p], dtype=np.float32)
        ev_w = np.exp(evs - evs.max())
        ev_w /= ev_w.sum()

        # Top=0, Mid=1, Bot=2, Discard=3
        target_map = {'Top': 0, 'Middle': 1, 'Bottom': 2}
        
        soft_labels = np.zeros((len(hand), NUM_CLASSES), dtype=np.float32)
        for p, w in zip(top_p, ev_w):
            d = p['d']
            p_dict = {}
            if p['p']:
                for part in p['p'].split(', '):
                    if part:
                        c, t = part.split('→')
                        p_dict[c] = t
            
            for i, hc in enumerate(hand):
                if hc == d:
                    soft_labels[i, 3] += w
                else:
                    t = p_dict.get(hc)
                    if t in target_map:
                        soft_labels[i, target_map[t]] += w

        rs = soft_labels.sum(axis=1, keepdims=True)
        rs = np.where(rs == 0, 1.0, rs)
        soft_labels /= rs

        best = placements[0]
        hard_labels = np.zeros(len(hand), dtype=np.int64)
        best_d = best['d']
        best_p = {}
        if best['p']:
            for part in best['p'].split(', '):
                if part:
                    c, t = part.split('→')
                    best_p[c] = t
                    
        for i, hc in enumerate(hand):
            if hc == best_d:
                hard_labels[i] = 3
            else:
                hard_labels[i] = target_map.get(best_p.get(hc), 3)

        return {
            'features': features,
            'soft_labels': soft_labels,
            'hard_labels': hard_labels,
            'best_ev': np.float32(best.get('ev', 0.0)),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'features': torch.from_numpy(s['features']),
            'soft_labels': torch.from_numpy(s['soft_labels']),
            'hard_labels': torch.from_numpy(s['hard_labels']),
            'best_ev': torch.tensor(s['best_ev']),
        }


class T1PlacementNet(nn.Module):
    """Transformer for T1 card placement."""

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
            nn.Linear(d_model, NUM_CLASSES),
        )
        self.ev_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, cards):
        # cards: (B, 8, CARD_DIM)
        x = self.card_embed(cards) + self.pos_embed[:, :cards.size(1)]
        x = self.encoder(x)
        # only predict for the hand cards (last 3 cards)
        hand_tokens = x[:, -3:, :]
        row_logits = self.row_head(hand_tokens)
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
        hard_loss = F.cross_entropy(logits.reshape(-1, NUM_CLASSES), hard_labels.reshape(-1))
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
        hard_loss = F.cross_entropy(logits.reshape(-1, NUM_CLASSES), hard_labels.reshape(-1))
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
    parser = argparse.ArgumentParser(description="Train T1 Placement Network")
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--weight-decay', type=float, default=0.02)
    parser.add_argument('--output', type=str, default='ai/models/t1_placement_net.pt')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--top-k', type=int, default=10)
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"T1 Config: d_model={args.d_model}, layers={args.num_layers}, "
          f"dropout={args.dropout}, wd={args.weight_decay}")

    all_samples = []
    with open(args.data, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_samples.append(json.loads(line.strip()))
    print(f"Total unique hands: {len(all_samples)}")

    random.seed(42)
    random.shuffle(all_samples)
    split = int(0.9 * len(all_samples))
    train_raw = all_samples[:split]
    val_raw = all_samples[split:]
    print(f"Train hands: {len(train_raw)}, Val hands: {len(val_raw)}")

    train_aug = augment_samples(train_raw)
    val_aug = augment_samples(val_raw)
    print(f"Train augmented: {len(train_aug)}, Val augmented: {len(val_aug)}")

    train_ds = T1PlacementDataset(train_aug, top_k=args.top_k)
    val_ds = T1PlacementDataset(val_aug, top_k=args.top_k)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = T1PlacementNet(
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
                    'n_params': n_params,
                    'version': 't1_v1',
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
