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

from ai.models.t1_network import (
    T1PlacementNet, CARD_DIM, NUM_CLASSES, encode_card_str, SUITS
)


def permute_card_str(c_str, suit_map):
    if c_str in ('JK', 'X1', 'X2'): return c_str
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
            hand_str = sample.get('hand', sample.get('hand_cards', ''))
            new_hand = [permute_card_str(c, suit_map) for c in hand_str.split()]
            hand_k = tuple(sorted(new_hand))
            if hand_k in seen:
                continue
            seen.add(hand_k)
            
            top, mid, bot = parse_board(sample['board'])
            new_top = " ".join([permute_card_str(c, suit_map) for c in top])
            new_mid = " ".join([permute_card_str(c, suit_map) for c in mid])
            new_bot = " ".join([permute_card_str(c, suit_map) for c in bot])
            new_board = f"Top[{new_top}] Mid[{new_mid}] Bot[{new_bot}]"
            
            # Opponent and dead cards
            opp_top = sample.get('opp_top', "").split()
            opp_mid = sample.get('opp_mid', "").split()
            opp_bot = sample.get('opp_bot', "").split()
            dead = sample.get('dead_cards', "").split()
            
            new_opp_top = " ".join([permute_card_str(c, suit_map) for c in opp_top])
            new_opp_mid = " ".join([permute_card_str(c, suit_map) for c in opp_mid])
            new_opp_bot = " ".join([permute_card_str(c, suit_map) for c in opp_bot])
            new_dead = " ".join([permute_card_str(c, suit_map) for c in dead])
            
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
                'opp_top': new_opp_top,
                'opp_mid': new_opp_mid,
                'opp_bot': new_opp_bot,
                'dead_cards': new_dead,
                'placements': new_placements,
                'original_ev': sample.get('original_ev', 0)
            })
        if len(augmented) % 100000 == 0:
            print(f"  Augmented {len(augmented)} samples...")
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
        hand_str = data.get('hand', data.get('hand_cards', ''))
        hand = hand_str.split()
        
        opp_top = data.get('opp_top', "").split()
        opp_mid = data.get('opp_mid', "").split()
        opp_bot = data.get('opp_bot', "").split()
        dead = data.get('dead_cards', "").split()
        
        all_cards = []
        for c in top: all_cards.append((c, 1))
        for c in mid: all_cards.append((c, 2))
        for c in bot: all_cards.append((c, 3))
        
        for c in opp_top: all_cards.append((c, 4))
        for c in opp_mid: all_cards.append((c, 5))
        for c in opp_bot: all_cards.append((c, 6))
        
        for c in dead: all_cards.append((c, 7))
        
        raw_features = [encode_card_str(c, r) for c, r in all_cards]
        from ai.models.t1_network import MAX_CARDS, CARD_DIM
        
        # Pad features with zeros
        features = np.zeros((MAX_CARDS, CARD_DIM), dtype=np.float32)
        features[:len(raw_features)] = np.stack(raw_features) if raw_features else np.zeros((0, CARD_DIM), dtype=np.float32)
        
        # Put hand cards exactly at the end of the sequence
        # hand length should be 3 for T1, but we use len(hand)
        features[-len(hand):] = np.stack([encode_card_str(c, 0) for c in hand])
        
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
    parser.add_argument('--position', type=str, default='all', choices=['all', 'independent', 'btn', 'bb'], help='Filter training data by position')
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"T1 Config: d_model={args.d_model}, layers={args.num_layers}, "
          f"dropout={args.dropout}, wd={args.weight_decay}")
    print(f"Position filter: {args.position}")

    all_samples = []
    with open(args.data, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line.strip())
            
            # Group placements by t0_idx
            t0_groups = {}
            if 't0_idx' in d and 't0_p' in d:
                t0_idx = d['t0_idx']
                t0_groups[t0_idx] = {'t0_p': d['t0_p'], 'placements': []}
                for p in d.get('placements', []):
                    p_str = p['p'].replace('->', '→')
                    t0_groups[t0_idx]['placements'].append({
                        'd': p['d'],
                        'p': p_str,
                        'ev': p['ev']
                    })
            else:
                for p in d.get('placements', []):
                    t0_idx = p['t0_idx']
                    if t0_idx not in t0_groups:
                        t0_groups[t0_idx] = {'t0_p': p['t0_p'], 'placements': []}
                    
                    p_str = p['p'].replace('->', '→')
                    t0_groups[t0_idx]['placements'].append({
                        'd': p['d'],
                        'p': p_str,
                        'ev': p['ev']
                    })
                
            for t0_idx, group in t0_groups.items():
                top, mid, bot = [], [], []
                for part in group['t0_p'].split(', '):
                    if not part: continue
                    c, dest = part.split('->')
                    if dest == 'Top': top.append(c)
                    elif dest == 'Middle': mid.append(c)
                    elif dest == 'Bottom': bot.append(c)
                board_str = f"Top[{' '.join(top)}] Mid[{' '.join(mid)}] Bot[{' '.join(bot)}]"
                
                sample = {
                    'board': board_str,
                    'hand': d['t1_hand'],
                    'opp_top': '',
                    'opp_mid': '',
                    'opp_bot': '',
                    'dead_cards': '',
                    'placements': group['placements'],
                    'position': 'independent'
                }
                
                if args.position != 'all' and sample.get('position', 'independent') != args.position:
                    continue
                all_samples.append(sample)
    print(f"Total unique boards: {len(all_samples)}")

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
                    'version': 't1_v2',
                    'position': args.position,
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
