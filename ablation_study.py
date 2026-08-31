#!/usr/bin/env python3
"""
Ablation study: systematically test improvements from the implementation plan.
Each experiment trains quickly (30 epochs) and reports val hand_acc + recall@K.
"""

import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from itertools import permutations
from collections import Counter
import time
import sys

SUITS = ['spades', 'hearts', 'diamonds', 'clubs']
CARD_DIM_BASE = 18  # 13 ranks + 4 suits + 1 joker
NUM_ROWS = 3


def encode_card_base(card):
    features = np.zeros(CARD_DIM_BASE, dtype=np.float32)
    rank = card.get('rank', '')
    suit = card.get('suit', '')
    if rank == 'Joker' or suit == 'joker':
        features[17] = 1.0
    else:
        rank_map = {'2':0,'3':1,'4':2,'5':3,'6':4,'7':5,
                    '8':6,'9':7,'T':8,'J':9,'Q':10,'K':11,'A':12}
        features[rank_map.get(rank, 0)] = 1.0
        suit_map = {'spades':0,'hearts':1,'diamonds':2,'clubs':3}
        features[13 + suit_map.get(suit, 0)] = 1.0
    return features


def encode_card_enhanced(card):
    """Enhanced encoding with hand-level features."""
    base = encode_card_base(card)
    # Extra features will be added at the hand level
    return base


def compute_hand_features(hand):
    """Compute hand-level features for all 5 cards.
    Returns (5, n_extra) array of extra features per card."""
    ranks = []
    suits = []
    rank_map = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,
                '8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14}
    
    for c in hand:
        r = c.get('rank', '')
        s = c.get('suit', '')
        ranks.append(rank_map.get(r, 0))
        suits.append(s)
    
    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)
    
    n_extra = 6  # pair_member, trip_member, flush_draw, straight_conn, high_card, royalty_potential
    extras = np.zeros((5, n_extra), dtype=np.float32)
    
    for i, (r, s) in enumerate(zip(ranks, suits)):
        # Is this card part of a pair?
        extras[i, 0] = 1.0 if rank_counts[r] >= 2 else 0.0
        # Part of trips?
        extras[i, 1] = 1.0 if rank_counts[r] >= 3 else 0.0
        # Flush draw (3+ same suit)
        extras[i, 2] = 1.0 if suit_counts[s] >= 3 else 0.0
        # Straight connectivity (adjacent rank within hand)
        has_adj = any(abs(r - r2) == 1 for j, r2 in enumerate(ranks) if j != i)
        extras[i, 3] = 1.0 if has_adj else 0.0
        # High card (T+)
        extras[i, 4] = 1.0 if r >= 10 else 0.0
        # Royalty potential (QQ+, pairs that score royalties in top)
        extras[i, 5] = 1.0 if r >= 12 and rank_counts[r] >= 2 else 0.0
    
    return extras


def card_key(c):
    return f"{c['rank']}_{c['suit']}"


def permute_card(card, suit_map):
    new = dict(card)
    s = card.get('suit', '')
    if s in suit_map:
        new['suit'] = suit_map[s]
    return new


def augment_samples(samples):
    all_perms = list(permutations(SUITS))
    augmented = []
    for sample in samples:
        seen = set()
        for perm in all_perms:
            suit_map = {SUITS[i]: perm[i] for i in range(4)}
            new_hand = [permute_card(c, suit_map) for c in sample['hand']]
            hand_key = tuple((c['rank'], c['suit']) for c in new_hand)
            if hand_key in seen:
                continue
            seen.add(hand_key)
            new_sample = dict(sample)
            new_sample['hand'] = new_hand
            sol = sample.get('solution', {})
            new_sample['solution'] = {
                'top': [permute_card(c, suit_map) for c in sol.get('top', [])],
                'mid': [permute_card(c, suit_map) for c in sol.get('mid', [])],
                'bot': [permute_card(c, suit_map) for c in sol.get('bot', [])],
            }
            new_placements = []
            for p in sample.get('placements', []):
                new_p = {
                    'top': [permute_card(c, suit_map) for c in p.get('top', [])],
                    'mid': [permute_card(c, suit_map) for c in p.get('mid', [])],
                    'bot': [permute_card(c, suit_map) for c in p.get('bot', [])],
                    'ev': p.get('ev', 0),
                }
                new_placements.append(new_p)
            new_sample['placements'] = new_placements
            augmented.append(new_sample)
    return augmented


class T0Dataset(Dataset):
    def __init__(self, samples, encode_fn, use_hand_features=False, top_k=10):
        self.data = []
        for s in samples:
            p = self._process(s, encode_fn, use_hand_features, top_k)
            if p:
                self.data.append(p)

    def _process(self, data, encode_fn, use_hand_features, top_k):
        hand = data.get('hand', [])
        if len(hand) != 5:
            return None
        features = np.stack([encode_fn(c) for c in hand])
        if use_hand_features:
            extras = compute_hand_features(hand)
            features = np.concatenate([features, extras], axis=1)
        
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
        return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]
        f = s['features'].copy()
        sl = s['soft_labels'].copy()
        hl = s['hard_labels'].copy()
        perm = np.random.permutation(5)
        f = f[perm]; sl = sl[perm]; hl = hl[perm]
        return {
            'features': torch.from_numpy(f),
            'soft_labels': torch.from_numpy(sl),
            'hard_labels': torch.from_numpy(hl),
            'best_ev': torch.tensor(s['best_ev']),
        }


class PlacementNet(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=4, dim_ff=256, dropout=0.1):
        super().__init__()
        self.card_embed = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, 5, d_model) * 0.02)
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

    def forward(self, x):
        h = self.card_embed(x) + self.pos_embed
        h = self.encoder(h)
        row_logits = self.row_head(h)
        ev_pred = self.ev_head(h.mean(dim=1))
        return row_logits, ev_pred


def train_and_eval(config, train_samples, val_samples, val_raw, device, max_epochs=50):
    """Train with given config and return val metrics."""
    encode_fn = config.get('encode_fn', encode_card_base)
    use_hf = config.get('use_hand_features', False)
    input_dim = CARD_DIM_BASE + (6 if use_hf else 0)
    
    train_ds = T0Dataset(train_samples, encode_fn, use_hf)
    val_ds = T0Dataset(val_samples, encode_fn, use_hf)
    
    train_loader = DataLoader(train_ds, batch_size=config.get('batch_size', 128), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)
    
    model = PlacementNet(
        input_dim=input_dim,
        d_model=config.get('d_model', 128),
        nhead=config.get('nhead', 4),
        num_layers=config.get('num_layers', 4),
        dim_ff=config.get('dim_ff', 256),
        dropout=config.get('dropout', 0.1),
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.get('lr', 3e-4),
                           weight_decay=config.get('weight_decay', 0.01))
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    best_val_hand = 0
    best_state = None
    patience = 0
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in train_loader:
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
        scheduler.step()
        
        # Val
        model.eval()
        val_hand = 0
        val_card = 0
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                hard_labels = batch['hard_labels'].to(device)
                logits, _ = model(features)
                pred = logits.argmax(dim=-1)
                val_card += (pred == hard_labels).float().mean().item()
                val_hand += (pred == hard_labels).float().prod(dim=1).mean().item()
                n += 1
        val_hand /= n
        val_card /= n
        
        if val_hand > best_val_hand:
            best_val_hand = val_hand
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 15:
                break
    
    # Evaluate recall with best model
    model.load_state_dict(best_state)
    model.eval()
    
    ks = [1, 5, 10, 20, 50]
    recall = {k: 0 for k in ks}
    total = 0
    
    with torch.no_grad():
        for sample in val_raw:
            hand = sample['hand']
            if len(hand) != 5:
                continue
            placements = sample.get('placements', [])
            if not placements:
                continue
            
            feats = np.stack([encode_fn(c) for c in hand])
            if use_hf:
                extras = compute_hand_features(hand)
                feats = np.concatenate([feats, extras], axis=1)
            feats_t = torch.from_numpy(feats).unsqueeze(0).to(device)
            logits, _ = model(feats_t)
            probs = torch.softmax(logits, dim=-1)[0]
            
            hand_keys = [card_key(c) for c in hand]
            scored = []
            for p in placements:
                cr = {}
                for rn, ri in [('top',0),('mid',1),('bot',2)]:
                    for card in p.get(rn,[]):
                        cr[card_key(card)] = ri
                score = sum(torch.log(probs[i, cr[key]] + 1e-8).item()
                           for i, key in enumerate(hand_keys) if key in cr)
                scored.append((score, p.get('ev', 0)))
            
            scored.sort(key=lambda x: x[0], reverse=True)
            best_ev = max(p.get('ev', -999) for p in placements)
            for rank, (sc, ev) in enumerate(scored, 1):
                if abs(ev - best_ev) < 0.001:
                    for k in recall:
                        if rank <= k:
                            recall[k] += 1
                    break
            total += 1
    
    recall_pct = {k: recall[k]/total*100 if total > 0 else 0 for k in ks}
    n_params = sum(p.numel() for p in model.parameters())
    
    return {
        'val_hand_acc': best_val_hand,
        'val_card_acc': val_card,
        'recall': recall_pct,
        'total_val': total,
        'n_params': n_params,
        'stopped_epoch': epoch,
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load data
    data_path = r'c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\t0_training_all.jsonl'
    all_samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_samples.append(json.loads(line.strip()))
    print(f"Total unique hands: {len(all_samples)}")
    
    random.seed(42)
    random.shuffle(all_samples)
    split = int(0.9 * len(all_samples))
    train_raw = all_samples[:split]
    val_raw = all_samples[split:]
    print(f"Train: {len(train_raw)}, Val: {len(val_raw)}")
    
    train_aug = augment_samples(train_raw)
    val_aug = augment_samples(val_raw)
    print(f"Train aug: {len(train_aug)}, Val aug: {len(val_aug)}")
    
    # Define experiments
    experiments = {
        'A_baseline': {
            'd_model': 128, 'num_layers': 4, 'dim_ff': 256,
            'dropout': 0.1, 'weight_decay': 0.01, 'lr': 3e-4,
        },
        'B_heavy_reg': {
            'd_model': 128, 'num_layers': 4, 'dim_ff': 256,
            'dropout': 0.3, 'weight_decay': 0.05, 'lr': 3e-4,
        },
        'C_small_model': {
            'd_model': 64, 'num_layers': 2, 'dim_ff': 128,
            'dropout': 0.2, 'weight_decay': 0.02, 'lr': 3e-4,
        },
        'D_hand_features': {
            'd_model': 128, 'num_layers': 4, 'dim_ff': 256,
            'dropout': 0.1, 'weight_decay': 0.01, 'lr': 3e-4,
            'use_hand_features': True,
        },
        'E_big_model': {
            'd_model': 256, 'num_layers': 6, 'dim_ff': 512,
            'dropout': 0.2, 'weight_decay': 0.02, 'lr': 1e-4,
        },
        'F_combined_best': {
            'd_model': 128, 'num_layers': 4, 'dim_ff': 256,
            'dropout': 0.2, 'weight_decay': 0.02, 'lr': 3e-4,
            'use_hand_features': True,
        },
    }
    
    results = {}
    for name, config in experiments.items():
        config['encode_fn'] = encode_card_base
        print(f"\n{'='*60}")
        print(f"Experiment: {name}")
        print(f"Config: d={config['d_model']} L={config['num_layers']} "
              f"drop={config['dropout']} wd={config['weight_decay']} "
              f"hf={config.get('use_hand_features', False)}")
        print(f"{'='*60}")
        
        t0 = time.time()
        res = train_and_eval(config, train_aug, val_aug, val_raw, device, max_epochs=60)
        elapsed = time.time() - t0
        results[name] = res
        
        print(f"  Val Hand Acc: {res['val_hand_acc']:.4f}")
        print(f"  Params: {res['n_params']:,}")
        print(f"  Stopped at epoch: {res['stopped_epoch']}")
        print(f"  Time: {elapsed:.0f}s")
        print(f"  Recall: ", end='')
        for k in sorted(res['recall']):
            print(f"@{k}={res['recall'][k]:.1f}% ", end='')
        print()
    
    # Summary table
    print(f"\n\n{'='*80}")
    print(f"{'ABLATION RESULTS':^80}")
    print(f"{'='*80}")
    print(f"{'Experiment':<20} {'HandAcc':>8} {'Params':>8} {'@1':>6} {'@5':>6} {'@10':>6} {'@20':>6} {'@50':>6}")
    print('-' * 80)
    for name, res in results.items():
        r = res['recall']
        print(f"{name:<20} {res['val_hand_acc']:>7.1%} {res['n_params']:>7,} "
              f"{r.get(1,0):>5.1f}% {r.get(5,0):>5.1f}% {r.get(10,0):>5.1f}% "
              f"{r.get(20,0):>5.1f}% {r.get(50,0):>5.1f}%")


if __name__ == '__main__':
    main()
