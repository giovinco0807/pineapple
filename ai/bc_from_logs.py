"""
BC Training from Game Logs (v2 - with augmentation)

Trains a policy network on human game data.
Each sample: (board_state, opponent_board, dealt_cards, turn) → placement_actions

Improvements over v1:
  - Suit permutation augmentation (up to 24x data)
  - Opponent board state included
  - Cosine LR schedule with warmup
  - Early stopping
"""
import sqlite3
import json
import sys
import os
import random
from pathlib import Path
from itertools import permutations

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Card encoding
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['h', 'd', 'c', 's']
CARD_FEATURES = 18  # 13 rank + 4 suit + 1 joker
SLOT_FEATURES = CARD_FEATURES + 1  # 19 (card + empty flag)

BOARD_SLOTS = 13  # top:3 + mid:5 + bot:5
MAX_DEALT = 5
NUM_TURNS = 5
NUM_ROWS = 4  # top, mid, bot, discard

# State = self_board(13*19) + opp_board(13*19) + dealt(5*19) + turn(5) + known_discards_count(1)
STATE_DIM = (BOARD_SLOTS * 2 + MAX_DEALT) * SLOT_FEATURES + NUM_TURNS + 1
ACTION_DIM = NUM_ROWS

# Pre-compute all suit permutations (4! = 24)
SUIT_PERMS = list(permutations(range(4)))


def encode_card(card_str):
    """Encode card string to feature vector."""
    feat = np.zeros(CARD_FEATURES, dtype=np.float32)
    if card_str in ('X1', 'X2', 'JK'):
        feat[17] = 1.0
    else:
        rank_ch = card_str[0]
        suit_ch = card_str[1]
        if rank_ch in RANKS:
            feat[RANKS.index(rank_ch)] = 1.0
        if suit_ch in SUITS:
            feat[13 + SUITS.index(suit_ch)] = 1.0
    return feat


def encode_slot(card_str=None):
    """Encode a board slot (card or empty)."""
    feat = np.zeros(SLOT_FEATURES, dtype=np.float32)
    if card_str is None:
        feat[18] = 1.0
    else:
        feat[:18] = encode_card(card_str)
    return feat


def permute_suit(card_str, perm):
    """Apply suit permutation to a card string."""
    if card_str in ('X1', 'X2', 'JK'):
        return card_str
    rank = card_str[0]
    suit_idx = SUITS.index(card_str[1])
    new_suit = SUITS[perm[suit_idx]]
    return rank + new_suit


def encode_state(board_self, board_opp, dealt_cards, turn, n_discards=0):
    """Encode full game state with opponent board."""
    state = np.zeros(STATE_DIM, dtype=np.float32)
    idx = 0
    
    # Self board: top(3) + mid(5) + bot(5)
    for row_name, max_cards in [('top', 3), ('middle', 5), ('bottom', 5)]:
        cards = board_self.get(row_name, [])
        for i in range(max_cards):
            if i < len(cards):
                state[idx:idx+SLOT_FEATURES] = encode_slot(cards[i])
            else:
                state[idx:idx+SLOT_FEATURES] = encode_slot(None)
            idx += SLOT_FEATURES
    
    # Opponent board: top(3) + mid(5) + bot(5)
    for row_name, max_cards in [('top', 3), ('middle', 5), ('bottom', 5)]:
        cards = board_opp.get(row_name, [])
        for i in range(max_cards):
            if i < len(cards):
                state[idx:idx+SLOT_FEATURES] = encode_slot(cards[i])
            else:
                state[idx:idx+SLOT_FEATURES] = encode_slot(None)
            idx += SLOT_FEATURES
    
    # Dealt cards (up to 5)
    for i in range(MAX_DEALT):
        if i < len(dealt_cards):
            state[idx:idx+SLOT_FEATURES] = encode_slot(dealt_cards[i])
        else:
            state[idx:idx+SLOT_FEATURES] = encode_slot(None)
        idx += SLOT_FEATURES
    
    # Turn number (one-hot)
    if 0 <= turn < NUM_TURNS:
        state[idx + turn] = 1.0
    idx += NUM_TURNS
    
    # Known discards count (normalized)
    state[idx] = min(n_discards / 10.0, 1.0)
    
    return state


def permute_board(board, perm):
    """Apply suit permutation to all cards in a board."""
    return {
        row: [permute_suit(c, perm) for c in cards]
        for row, cards in board.items()
    }


class GameLogDataset(Dataset):
    """Dataset from SQLite game logs with suit permutation augmentation."""
    
    def __init__(self, db_path, augment=True, max_perms=6):
        """
        Args:
            db_path: Path to SQLite database
            augment: Whether to use suit permutation augmentation
            max_perms: Max number of permutations per sample (6 keeps ~6x data)
        """
        self.samples = []
        row_map = {'top': 0, 'middle': 1, 'bottom': 2}
        
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT board_self, board_opponent, dealt_cards, 
                   action_placements, action_discard, turn, known_discards
            FROM turns 
            WHERE action_placements IS NOT NULL
        """)
        
        raw_samples = []
        for row in cur.fetchall():
            board_self = json.loads(row[0])
            board_opp = json.loads(row[1])
            dealt = json.loads(row[2])
            placements = json.loads(row[3])
            discard = row[4]
            turn = row[5]
            known_disc = json.loads(row[6]) if row[6] else []
            
            # Build action labels
            card_actions = []
            for card in dealt:
                action = 3  # discard
                for placement in placements:
                    if placement[0] == card:
                        action = row_map.get(placement[1], 3)
                        break
                card_actions.append(action)
            
            raw_samples.append({
                'board_self': board_self,
                'board_opp': board_opp,
                'dealt': dealt,
                'turn': turn,
                'actions': card_actions,
                'n_discards': len(known_disc),
            })
        
        conn.close()
        print(f"Raw samples: {len(raw_samples)}")
        
        # Generate augmented samples
        if augment:
            perms_to_use = random.sample(SUIT_PERMS, min(max_perms, len(SUIT_PERMS)))
        else:
            perms_to_use = [(0, 1, 2, 3)]  # identity only
        
        for sample in raw_samples:
            for perm in perms_to_use:
                board_s = permute_board(sample['board_self'], perm)
                board_o = permute_board(sample['board_opp'], perm)
                dealt_p = [permute_suit(c, perm) for c in sample['dealt']]
                
                state = encode_state(board_s, board_o, dealt_p, 
                                     sample['turn'], sample['n_discards'])
                
                actions = np.array(sample['actions'][:MAX_DEALT], dtype=np.int64)
                # Pad
                padded = np.full(MAX_DEALT, -1, dtype=np.int64)
                padded[:len(actions)] = actions
                
                self.samples.append({
                    'state': state,
                    'actions': padded,
                    'n_cards': len(sample['dealt']),
                })
        
        print(f"Augmented samples: {len(self.samples)} ({len(self.samples)/len(raw_samples):.1f}x)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'state': torch.tensor(s['state'], dtype=torch.float32),
            'actions': torch.tensor(s['actions'], dtype=torch.long),
            'n_cards': s['n_cards'],
        }


class PlacementNet(nn.Module):
    """Predicts placement row for each dealt card."""
    
    def __init__(self, state_dim=STATE_DIM, hidden=256, num_layers=4):
        super().__init__()
        layers = []
        in_dim = state_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(0.15),
            ])
            in_dim = hidden
        self.shared = nn.Sequential(*layers)
        self.card_heads = nn.ModuleList([
            nn.Linear(hidden, ACTION_DIM) for _ in range(MAX_DEALT)
        ])
    
    def forward(self, state):
        h = self.shared(state)
        logits = torch.stack([head(h) for head in self.card_heads], dim=1)
        return logits


def train(db_path, epochs=100, batch_size=64, lr=1e-3, output_dir='ai/models/bc_game',
          augment=True, max_perms=6, patience=15):
    dataset = GameLogDataset(db_path, augment=augment, max_perms=max_perms)
    
    # Split train/val (90/10)
    n = len(dataset)
    n_val = max(1, n // 10)
    n_train = n - n_val
    
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val], generator=gen)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)
    
    model = PlacementNet()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Cosine annealing with warmup
    warmup_epochs = 5
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-5)
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")
    print(f"State dim: {STATE_DIM}")
    print(f"Training: {n_train}, Validation: {n_val}")
    print(f"Device: {device}")
    print()
    
    best_val_acc = 0
    no_improve = 0
    
    for epoch in range(epochs):
        # Warmup LR
        if epoch < warmup_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = lr * (epoch + 1) / warmup_epochs
        
        # Train
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            states = batch['state'].to(device)
            actions = batch['actions'].to(device)
            
            logits = model(states)
            loss = criterion(logits.view(-1, ACTION_DIM), actions.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=2)
            mask = actions >= 0
            correct += ((preds == actions) & mask).sum().item()
            total += mask.sum().item()
        
        if epoch >= warmup_epochs:
            scheduler.step()
        
        train_acc = correct / total if total > 0 else 0
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                states = batch['state'].to(device)
                actions = batch['actions'].to(device)
                
                logits = model(states)
                loss = criterion(logits.view(-1, ACTION_DIM), actions.view(-1))
                val_loss += loss.item()
                
                preds = logits.argmax(dim=2)
                mask = actions >= 0
                val_correct += ((preds == actions) & mask).sum().item()
                val_total += mask.sum().item()
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        cur_lr = optimizer.param_groups[0]['lr']
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}: loss={total_loss/len(train_loader):.4f} "
                  f"train={train_acc*100:.1f}% val={val_acc*100:.1f}% lr={cur_lr:.1e}")
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            os.makedirs(output_dir, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'state_dim': STATE_DIM,
                'val_acc': val_acc,
                'epoch': epoch + 1,
            }, os.path.join(output_dir, 'best_model.pt'))
        else:
            no_improve += 1
        
        # Early stopping
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
    
    print(f"\nBest validation accuracy: {best_val_acc*100:.1f}%")
    
    # Save final
    torch.save({
        'model_state_dict': model.state_dict(),
        'state_dim': STATE_DIM,
        'val_acc': val_acc,
        'epoch': epochs,
    }, os.path.join(output_dir, 'final_model.pt'))
    print(f"Models saved to {output_dir}/")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='data/ofc_logs.db')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output', default='ai/models/bc_game')
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--max-perms', type=int, default=6, help='Max suit permutations (1-24)')
    parser.add_argument('--patience', type=int, default=15)
    args = parser.parse_args()
    
    train(args.db, args.epochs, args.batch_size, args.lr, args.output,
          augment=not args.no_augment, max_perms=args.max_perms, patience=args.patience)
