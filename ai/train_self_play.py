#!/usr/bin/env python3
"""
Continuous Training Script for Self-Play Data.
Trains a model (e.g. t1_bb) using MCTS visit probabilities from self_play_data.jsonl
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

from ai.models.t1_network import (
    T1PlacementNet, CARD_DIM, NUM_CLASSES, encode_card_str, SUITS, MAX_CARDS
)

def parse_board(board_str):
    import re
    top_match = re.search(r'Top\[(.*?)\]', board_str)
    mid_match = re.search(r'Mid\[(.*?)\]', board_str)
    bot_match = re.search(r'Bot\[(.*?)\]', board_str)
    top = top_match.group(1).split() if top_match and top_match.group(1) else []
    mid = mid_match.group(1).split() if mid_match and mid_match.group(1) else []
    bot = bot_match.group(1).split() if bot_match and bot_match.group(1) else []
    return top, mid, bot

class SelfPlayDataset(Dataset):
    def __init__(self, samples):
        self.samples = []
        for s in samples:
            p = self._process(s)
            if p:
                self.samples.append(p)
        print(f"  Dataset: {len(self.samples)} samples")

    def _process(self, data):
        top, mid, bot = parse_board(data['board'])
        hand = data.get('hand', '').split()
        
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
        
        features = np.zeros((MAX_CARDS, CARD_DIM), dtype=np.float32)
        features[:len(raw_features)] = np.stack(raw_features) if raw_features else np.zeros((0, CARD_DIM), dtype=np.float32)
        if hand:
            features[-len(hand):] = np.stack([encode_card_str(c, 0) for c in hand])
        
        placements = data.get('placements', [])
        if not placements:
            return None

        # Top=0, Mid=1, Bot=2, Discard=3
        target_map = {'Top': 0, 'Middle': 1, 'Bottom': 2}
        
        soft_labels = np.zeros((len(hand), NUM_CLASSES), dtype=np.float32)
        
        # In self-play, visit_prob is already normalized MCTS probability
        for p in placements:
            w = p['visit_prob']
            if w <= 0.0: continue
            
            d = p['d']
            p_dict = {}
            if p['p']:
                for part in p['p'].split(', '):
                    if part:
                        c, t = part.split('->')
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

        # Hard labels from best action (max visit_prob)
        best = max(placements, key=lambda x: x['visit_prob'])
        hard_labels = np.zeros(len(hand), dtype=np.int64)
        best_d = best['d']
        best_p = {}
        if best['p']:
            for part in best['p'].split(', '):
                if part:
                    c, t = part.split('->')
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
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'features': torch.from_numpy(s['features']),
            'soft_labels': torch.from_numpy(s['soft_labels']),
            'hard_labels': torch.from_numpy(s['hard_labels']),
        }

def train_epoch(model, loader, optimizer, device, n_hand):
    model.train()
    total_loss = total_card = total_hand = 0
    for batch in loader:
        features = batch['features'].to(device)
        soft_labels = batch['soft_labels'].to(device)
        hard_labels = batch['hard_labels'].to(device)
        
        optimizer.zero_grad()
        logits, _ = model(features, n_hand=n_hand)
        
        log_probs = F.log_softmax(logits, dim=-1)
        soft_loss = -(soft_labels * log_probs).sum(dim=-1).mean()
        hard_loss = F.cross_entropy(logits.reshape(-1, NUM_CLASSES), hard_labels.reshape(-1))
        
        loss = 0.7 * soft_loss + 0.3 * hard_loss
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='rust_solver/self_play/self_play_data.jsonl')
    parser.add_argument('--model-name', type=str, default='t1_bb', help='Model filter (e.g. t1_bb, t0_btn)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--output', type=str, default='ai/models/t1_placement_net_bb.pt')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"Loading {args.data} for model {args.model_name}")
    all_samples = []
    with open(args.data, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line.strip())
            if d.get('model') == args.model_name:
                all_samples.append(d)
                
    if not all_samples:
        print("No samples found. Exiting.")
        return
        
    print(f"Found {len(all_samples)} samples.")

    # Load existing model if it exists
    model = T1PlacementNet(d_model=128, nhead=4, num_layers=4, dim_ff=256, dropout=0.0).to(args.device)
    if Path(args.output).exists():
        checkpoint = torch.load(args.output, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded existing model weights.")

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    train_ds = SelfPlayDataset(all_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    n_hand = 5 if args.model_name.startswith('t0') else 3
    for epoch in range(1, args.epochs + 1):
        loss, card_acc, hand_acc = train_epoch(model, train_loader, optimizer, args.device, n_hand)
        print(f"Epoch {epoch:2d}/{args.epochs} | Loss={loss:.4f} CardAcc={card_acc:.4f} HandAcc={hand_acc:.4f}")
        
    # Save updated model
    torch.save({
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'val_card_acc': card_acc,
        'val_hand_acc': hand_acc,
        'config': {'d_model': 128, 'num_layers': 4, 'dropout': 0.0}
    }, args.output)
    print(f"Model saved to {args.output}")

if __name__ == '__main__':
    main()
