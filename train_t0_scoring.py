#!/usr/bin/env python3
"""
T0 Scoring Model: Predict EV for a given (hand, placement) pair.

Instead of classifying cards into rows, this model scores each candidate
placement and ranks them by predicted EV. This directly optimizes for recall@K.

Input:  5 cards × (rank[13] + suit[4] + joker[1] + row[3]) = 5 × 21
Output: Predicted EV (scalar)

Training: Each hand × 100 placements = 100× more training signal than classifier.
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


CARD_DIM = 21  # 13 ranks + 4 suits + 1 joker + 3 rows
NUM_ROWS = 3
SUITS = ['spades', 'hearts', 'diamonds', 'clubs']


def encode_card_with_row(card: dict, row: int) -> np.ndarray:
    """Encode a card with its row assignment."""
    features = np.zeros(CARD_DIM, dtype=np.float32)
    rank = card.get('rank', '')
    suit = card.get('suit', '')

    if rank == 'Joker' or suit == 'joker':
        features[17] = 1.0  # Joker flag
    else:
        rank_map = {'2':0,'3':1,'4':2,'5':3,'6':4,'7':5,
                    '8':6,'9':7,'T':8,'J':9,'Q':10,'K':11,'A':12}
        features[rank_map.get(rank, 0)] = 1.0
        suit_map = {'spades':0,'hearts':1,'diamonds':2,'clubs':3}
        features[13 + suit_map.get(suit, 0)] = 1.0

    # Row assignment (one-hot)
    features[18 + row] = 1.0
    return features


def card_key(c):
    return f"{c['rank']}_{c['suit']}"


def permute_card(card, suit_map):
    new = dict(card)
    s = card.get('suit', '')
    if s in suit_map:
        new['suit'] = suit_map[s]
    return new


def augment_hand_placements(hand, placements, best_ev):
    """Generate suit-augmented versions of (hand, placements) pairs."""
    all_perms = list(permutations(SUITS))
    results = []
    seen = set()

    for perm in all_perms:
        suit_map = {SUITS[i]: perm[i] for i in range(4)}
        new_hand = [permute_card(c, suit_map) for c in hand]
        hand_key_tuple = tuple((c['rank'], c['suit']) for c in new_hand)
        if hand_key_tuple in seen:
            continue
        seen.add(hand_key_tuple)

        new_placements = []
        for p in placements:
            new_p = {
                'top': [permute_card(c, suit_map) for c in p.get('top', [])],
                'mid': [permute_card(c, suit_map) for c in p.get('mid', [])],
                'bot': [permute_card(c, suit_map) for c in p.get('bot', [])],
                'ev': p.get('ev', 0),
            }
            new_placements.append(new_p)
        results.append((new_hand, new_placements, best_ev))

    return results


class T0ScoringDataset(Dataset):
    """Each sample is (hand_with_placement, ev)."""

    def __init__(self, hand_placement_pairs, augment_order=True):
        self.augment_order = augment_order
        self.data = hand_placement_pairs  # list of (features[5,21], ev)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, ev = self.data[idx]
        f = features.copy()
        if self.augment_order:
            perm = np.random.permutation(5)
            f = f[perm]
        return torch.from_numpy(f), torch.tensor(ev, dtype=torch.float32)


class T0ScoringNet(nn.Module):
    """Score a (hand, placement) pair → predicted EV."""

    def __init__(self, d_model=128, nhead=4, num_layers=4, dim_ff=256):
        super().__init__()
        self.card_embed = nn.Sequential(
            nn.Linear(CARD_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, 5, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        """x: (B, 5, 21) → (B, 1)"""
        h = self.card_embed(x)  # (B, 5, d_model)
        h = h + self.pos_embed
        h = self.encoder(h)  # (B, 5, d_model)
        h = h.mean(dim=1)  # (B, d_model) global pool
        return self.head(h).squeeze(-1)  # (B,)


def build_pairs(hand, placements):
    """Build (features, ev) pairs from a hand and its placements."""
    hand_keys = [card_key(c) for c in hand]
    pairs = []

    for p in placements:
        card_row = {}
        for row_name, row_idx in [('top', 0), ('mid', 1), ('bot', 2)]:
            for card in p.get(row_name, []):
                card_row[card_key(card)] = row_idx

        features = np.zeros((5, CARD_DIM), dtype=np.float32)
        for i, (card, key) in enumerate(zip(hand, hand_keys)):
            row = card_row.get(key, 2)  # default bot
            features[i] = encode_card_with_row(card, row)

        pairs.append((features, p.get('ev', 0.0)))

    return pairs


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load data
    data_path = r'c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\t0_training_v3.jsonl'
    all_samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_samples.append(json.loads(line.strip()))
    print(f"Total unique hands: {len(all_samples)}")

    # Split by unique hands BEFORE augmentation
    random.seed(42)
    random.shuffle(all_samples)
    split = int(0.9 * len(all_samples))
    train_raw = all_samples[:split]
    val_raw = all_samples[split:]
    print(f"Train unique hands: {len(train_raw)}")
    print(f"Val unique hands: {len(val_raw)}")

    # Build training pairs with suit augmentation
    print("Building training pairs with suit augmentation...")
    train_pairs = []
    for sample in train_raw:
        hand = sample['hand']
        placements = sample.get('placements', [])
        best_ev = sample.get('best_ev', 0)
        if len(hand) != 5 or not placements:
            continue
        for aug_hand, aug_placements, _ in augment_hand_placements(hand, placements, best_ev):
            train_pairs.extend(build_pairs(aug_hand, aug_placements))

    # Validation pairs (also augmented for consistent eval)
    val_pairs = []
    for sample in val_raw:
        hand = sample['hand']
        placements = sample.get('placements', [])
        if len(hand) != 5 or not placements:
            continue
        for aug_hand, aug_placements, _ in augment_hand_placements(hand, placements, sample.get('best_ev', 0)):
            val_pairs.extend(build_pairs(aug_hand, aug_placements))

    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")

    # Normalize EV targets
    all_evs = [ev for _, ev in train_pairs]
    ev_mean = np.mean(all_evs)
    ev_std = np.std(all_evs) + 1e-6
    print(f"EV stats: mean={ev_mean:.2f}, std={ev_std:.2f}")

    train_pairs_norm = [(f, (ev - ev_mean) / ev_std) for f, ev in train_pairs]
    val_pairs_norm = [(f, (ev - ev_mean) / ev_std) for f, ev in val_pairs]

    train_ds = T0ScoringDataset(train_pairs_norm, augment_order=True)
    val_ds = T0ScoringDataset(val_pairs_norm, augment_order=False)

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)

    # Model
    model = T0ScoringNet(d_model=128, nhead=4, num_layers=4, dim_ff=256).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)

    best_val_loss = float('inf')
    patience = 0
    max_patience = 20
    output_path = 'ai/models/t0_scoring_net_v1.pt'

    for epoch in range(1, 151):
        # Train
        model.train()
        train_loss = 0
        for feats, evs in train_loader:
            feats, evs = feats.to(device), evs.to(device)
            optimizer.zero_grad()
            pred = model(feats)
            loss = F.mse_loss(pred, evs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for feats, evs in val_loader:
                feats, evs = feats.to(device), evs.to(device)
                pred = model(feats)
                val_loss += F.mse_loss(pred, evs).item()
        val_loss /= len(val_loader)
        scheduler.step()

        if epoch % 5 == 0 or epoch <= 5:
            print(f"Epoch {epoch:3d}/150 | Train MSE={train_loss:.4f} | Val MSE={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'ev_mean': ev_mean,
                'ev_std': ev_std,
                'config': {'d_model': 128, 'num_layers': 4, 'n_params': n_params},
            }, output_path)
            if epoch % 5 == 0 or epoch <= 3:
                print(f"  >>> Saved best (val_loss={val_loss:.4f})")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nTraining complete. Best val MSE: {best_val_loss:.4f}")

    # Evaluate recall@K on UNSEEN hands (non-augmented)
    ckpt = torch.load(output_path, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    ks = [1, 3, 5, 10, 20, 50, 100]
    recall = {k: 0 for k in ks}
    total = 0
    ev_errors = []

    with torch.no_grad():
        for sample in val_raw:
            hand = sample['hand']
            placements = sample.get('placements', [])
            if len(hand) != 5 or not placements:
                continue

            # Score all placements
            hand_keys = [card_key(c) for c in hand]
            all_feats = []
            all_evs = []

            for p in placements:
                card_row = {}
                for rn, ri in [('top', 0), ('mid', 1), ('bot', 2)]:
                    for card in p.get(rn, []):
                        card_row[card_key(card)] = ri
                features = np.zeros((5, CARD_DIM), dtype=np.float32)
                for i, (card, key) in enumerate(zip(hand, hand_keys)):
                    features[i] = encode_card_with_row(card, card_row.get(key, 2))
                all_feats.append(features)
                all_evs.append(p.get('ev', 0))

            feats_batch = torch.from_numpy(np.stack(all_feats)).to(device)
            pred_scores = model(feats_batch).cpu().numpy()

            # Rank by predicted score
            ranked_indices = np.argsort(-pred_scores)
            best_ev = max(all_evs)

            for rank, idx in enumerate(ranked_indices, 1):
                if abs(all_evs[idx] - best_ev) < 0.001:
                    for k in recall:
                        if rank <= k:
                            recall[k] += 1
                    break

            # Track EV prediction error
            best_idx = np.argmax(all_evs)
            pred_best_idx = ranked_indices[0]
            ev_errors.append(all_evs[best_idx] - all_evs[pred_best_idx])
            total += 1

    print(f"\n{'='*55}")
    print(f"RECALL on UNSEEN hands ({total} unique hands)")
    print(f"{'='*55}")
    for k in sorted(recall):
        pct = recall[k] / total * 100
        print(f"  @{k:>3}: {recall[k]:>4}/{total} = {pct:.1f}%")

    print(f"\nEV loss when picking model's #1:")
    print(f"  Mean: {np.mean(ev_errors):.3f}")
    print(f"  Median: {np.median(ev_errors):.3f}")
    print(f"  Max: {np.max(ev_errors):.3f}")
    print(f"  Model picks true best: {sum(1 for e in ev_errors if e < 0.001)}/{total} = {sum(1 for e in ev_errors if e < 0.001)/total*100:.1f}%")

    print(f"\nModel saved to: {output_path}")


if __name__ == '__main__':
    main()
