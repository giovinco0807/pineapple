#!/usr/bin/env python3
"""
Train T0 with PROPER validation: split by unique hands BEFORE suit augmentation.
This prevents data leakage from suit-permuted duplicates appearing in both splits.
"""

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

import sys
sys.path.insert(0, 'ai')
from train_t0_placement import T0PlacementNet, encode_card, CARD_DIM, NUM_ROWS, MAX_CARDS


SUITS = ['spades', 'hearts', 'diamonds', 'clubs']


def card_key(c):
    return f"{c['rank']}_{c['suit']}"


def permute_card(card, suit_map):
    new = dict(card)
    s = card.get('suit', '')
    if s in suit_map:
        new['suit'] = suit_map[s]
    return new


def permute_sample(sample, suit_map):
    """Apply suit permutation to a sample."""
    new = dict(sample)
    new['hand'] = [permute_card(c, suit_map) for c in sample['hand']]
    sol = sample.get('solution', {})
    new['solution'] = {
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
    new['placements'] = new_placements
    return new


def augment_samples(samples):
    """Apply all 24 suit permutations to each sample, deduplicating."""
    all_perms = list(permutations(SUITS))
    augmented = []
    for sample in samples:
        seen = set()
        for perm in all_perms:
            suit_map = {SUITS[i]: perm[i] for i in range(4)}
            new_sample = permute_sample(sample, suit_map)
            hand_key = tuple((c['rank'], c['suit']) for c in new_sample['hand'])
            if hand_key not in seen:
                seen.add(hand_key)
                augmented.append(new_sample)
    return augmented


class T0Dataset(Dataset):
    def __init__(self, samples, augment_order=True, top_k=10):
        self.augment_order = augment_order
        self.top_k = top_k
        self.data = []
        for s in samples:
            processed = self._process(s)
            if processed:
                self.data.append(processed)
    
    def _process(self, data):
        hand = data.get('hand', [])
        if len(hand) != 5:
            return None
        features = np.stack([encode_card(c) for c in hand])
        placements = data.get('placements', [])
        if not placements:
            return None
        
        placements.sort(key=lambda x: x.get('ev', -999), reverse=True)
        top_placements = placements[:self.top_k]
        evs = np.array([p['ev'] for p in top_placements], dtype=np.float32)
        ev_weights = np.exp(evs - evs.max())
        ev_weights /= ev_weights.sum()
        
        soft_labels = np.zeros((5, NUM_ROWS), dtype=np.float32)
        hand_keys = [card_key(c) for c in hand]
        
        for p, w in zip(top_placements, ev_weights):
            card_row = {}
            for row_name, row_idx in [('top', 0), ('mid', 1), ('bot', 2)]:
                for card in p.get(row_name, []):
                    card_row[card_key(card)] = row_idx
            for i, key in enumerate(hand_keys):
                if key in card_row:
                    soft_labels[i, card_row[key]] += w
        
        row_sums = soft_labels.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        soft_labels /= row_sums
        
        best = placements[0]
        hard_labels = np.zeros(5, dtype=np.int64)
        best_card_row = {}
        for row_name, row_idx in [('top', 0), ('mid', 1), ('bot', 2)]:
            for card in best.get(row_name, []):
                best_card_row[card_key(card)] = row_idx
        for i, key in enumerate(hand_keys):
            hard_labels[i] = best_card_row.get(key, 2)
        
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
        features = s['features'].copy()
        soft_labels = s['soft_labels'].copy()
        hard_labels = s['hard_labels'].copy()
        if self.augment_order:
            perm = np.random.permutation(5)
            features = features[perm]
            soft_labels = soft_labels[perm]
            hard_labels = hard_labels[perm]
        return {
            'features': torch.from_numpy(features),
            'soft_labels': torch.from_numpy(soft_labels),
            'hard_labels': torch.from_numpy(hard_labels),
            'best_ev': torch.tensor(s['best_ev']),
        }


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_card_acc = 0
    total_hand_acc = 0
    for batch in loader:
        features = batch['features'].to(device)
        soft_labels = batch['soft_labels'].to(device)
        hard_labels = batch['hard_labels'].to(device)
        best_ev = batch['best_ev'].to(device)
        optimizer.zero_grad()
        row_logits, ev_pred = model(features)
        log_probs = F.log_softmax(row_logits, dim=-1)
        soft_loss = -(soft_labels * log_probs).sum(dim=-1).mean()
        hard_loss = F.cross_entropy(row_logits.reshape(-1, NUM_ROWS), hard_labels.reshape(-1))
        ev_loss = F.mse_loss(ev_pred.squeeze(-1), best_ev)
        loss = 0.7 * soft_loss + 0.3 * hard_loss + 0.1 * ev_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        pred = row_logits.argmax(dim=-1)
        card_acc = (pred == hard_labels).float().mean().item()
        hand_acc = (pred == hard_labels).float().prod(dim=1).mean().item()
        total_card_acc += card_acc
        total_hand_acc += hand_acc
    n = len(loader)
    return total_loss/n, total_card_acc/n, total_hand_acc/n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    total_card_acc = 0
    total_hand_acc = 0
    for batch in loader:
        features = batch['features'].to(device)
        soft_labels = batch['soft_labels'].to(device)
        hard_labels = batch['hard_labels'].to(device)
        best_ev = batch['best_ev'].to(device)
        row_logits, ev_pred = model(features)
        log_probs = F.log_softmax(row_logits, dim=-1)
        soft_loss = -(soft_labels * log_probs).sum(dim=-1).mean()
        hard_loss = F.cross_entropy(row_logits.reshape(-1, NUM_ROWS), hard_labels.reshape(-1))
        ev_loss = F.mse_loss(ev_pred.squeeze(-1), best_ev)
        loss = 0.7 * soft_loss + 0.3 * hard_loss + 0.1 * ev_loss
        total_loss += loss.item()
        pred = row_logits.argmax(dim=-1)
        card_acc = (pred == hard_labels).float().mean().item()
        hand_acc = (pred == hard_labels).float().prod(dim=1).mean().item()
        total_card_acc += card_acc
        total_hand_acc += hand_acc
    n = len(loader)
    return total_loss/n, total_card_acc/n, total_hand_acc/n


@torch.no_grad()
def evaluate_recall(model, val_samples_raw, device):
    """Evaluate recall@K on raw validation samples (before augmentation)."""
    model.eval()
    ks = [1, 3, 5, 10, 20, 50]
    recall = {k: 0 for k in ks}
    total = 0
    
    for sample in val_samples_raw:
        hand = sample['hand']
        if len(hand) != 5:
            continue
        placements = sample.get('placements', [])
        if not placements:
            continue
        
        feats = np.stack([encode_card(c) for c in hand])
        feats_t = torch.from_numpy(feats).unsqueeze(0).to(device)
        logits, _ = model(feats_t)
        probs = torch.softmax(logits, dim=-1)[0]
        
        hand_keys = [card_key(c) for c in hand]
        scored = []
        for p in placements:
            cr = {}
            for rn, ri in [('top', 0), ('mid', 1), ('bot', 2)]:
                for card in p.get(rn, []):
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
    
    return recall, total


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load all unique hands
    data_path = r'c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\t0_training_all.jsonl'
    all_samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_samples.append(json.loads(line.strip()))
    
    print(f"Total unique hands: {len(all_samples)}")
    
    # CRITICAL: Split by unique hands BEFORE augmentation
    random.seed(42)
    random.shuffle(all_samples)
    split = int(0.9 * len(all_samples))
    train_raw = all_samples[:split]
    val_raw = all_samples[split:]
    
    print(f"Train unique hands: {len(train_raw)}")
    print(f"Val unique hands: {len(val_raw)} (COMPLETELY UNSEEN rank patterns)")
    
    # Augment ONLY training data
    train_aug = augment_samples(train_raw)
    val_aug = augment_samples(val_raw)
    
    print(f"Train after augmentation: {len(train_aug)}")
    print(f"Val after augmentation: {len(val_aug)}")
    
    # Create datasets
    train_ds = T0Dataset(train_aug, augment_order=True)
    val_ds = T0Dataset(val_aug, augment_order=False)
    
    print(f"Train dataset: {len(train_ds)}")
    print(f"Val dataset: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)
    
    # Model
    model = T0PlacementNet(d_model=128, nhead=4, num_layers=4, dim_ff=256).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    
    best_val_hand_acc = 0
    patience = 0
    max_patience = 30
    output_path = 'ai/models/t0_placement_net_v3_aug_proper.pt'
    
    for epoch in range(1, 201):
        train_loss, train_card, train_hand = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_card, val_hand = evaluate(model, val_loader, device)
        scheduler.step()
        
        if epoch % 5 == 0 or epoch <= 5:
            print(f"Epoch {epoch:3d}/200 | "
                  f"Train L={train_loss:.4f} Card={train_card:.3f} Hand={train_hand:.3f} | "
                  f"Val L={val_loss:.4f} Card={val_card:.3f} Hand={val_hand:.3f}")
        
        if val_hand > best_val_hand_acc:
            best_val_hand_acc = val_hand
            patience = 0
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_card_acc': val_card,
                'val_hand_acc': val_hand,
                'config': {'d_model': 128, 'num_layers': 4, 'n_params': n_params},
            }, output_path)
            if epoch % 5 == 0 or epoch <= 5:
                print(f"  >>> Saved best (hand_acc={val_hand:.4f})")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    print(f"\nTraining complete. Best val hand accuracy: {best_val_hand_acc:.4f}")
    
    # Load best model and evaluate recall on UNSEEN val hands
    ckpt = torch.load(output_path, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    
    recall, total = evaluate_recall(model, val_raw, device)
    
    print(f"\n{'='*50}")
    print(f"RECALL on UNSEEN hands ({total} unique hands)")
    print(f"{'='*50}")
    for k in sorted(recall):
        pct = recall[k] / total * 100
        print(f"  @{k:>2}: {recall[k]:>4}/{total} = {pct:.1f}%")
    
    print(f"\nModel saved to: {output_path}")


if __name__ == '__main__':
    main()
