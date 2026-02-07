"""
Training script for Improved Policy Network (Card Ranking Approach).

Models each card's suitability for Top placement individually.
Much simpler than 364-class classification.

Usage:
    python train_policy_v2.py --data ai/data/fl_all.jsonl --model transformer --epochs 100
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

import sys
sys.path.insert(0, str(Path(__file__).parent))

from policy_network_v2 import (
    CardRankingNetwork,
    TransformerPolicyNetwork,
    PairwiseRankingLoss,
    CARD_DIM
)


class CardRankingDataset(Dataset):
    """Dataset for card ranking training."""
    
    def __init__(self, data_path: str, max_cards: int = 17, augment: bool = True):
        self.max_cards = max_cards
        self.augment = augment
        self.samples = []
        
        print(f"Loading data from {data_path}...")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading"):
                try:
                    data = json.loads(line)
                    sample = self._process_sample(data)
                    if sample:
                        self.samples.append(sample)
                except Exception as e:
                    continue
        
        print(f"Loaded {len(self.samples)} samples")
    
    def _encode_card(self, c: dict) -> np.ndarray:
        """Encode a single card."""
        features = np.zeros(CARD_DIM, dtype=np.float32)
        
        rank = c.get('rank', '')
        suit = c.get('suit', '')
        is_joker = rank == 'Joker' or suit == 'joker'
        
        if is_joker:
            features[CARD_DIM - 1] = 1.0  # Joker flag
        else:
            # Rank
            rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5,
                       '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
            rank_idx = rank_map.get(rank, 0)
            features[rank_idx] = 1.0
            
            # Suit
            suit_map = {'spades': 0, 'hearts': 1, 'diamonds': 2, 'clubs': 3}
            suit_idx = suit_map.get(suit, 0)
            features[13 + suit_idx] = 1.0
        
        return features
    
    def _process_sample(self, data: dict) -> dict:
        """Process a single sample."""
        hand_data = data.get('hand', [])
        n_cards = len(hand_data)
        
        if n_cards < 13 or n_cards > 17:
            return None
        
        # Encode hand
        hand_features = np.zeros((self.max_cards, CARD_DIM), dtype=np.float32)
        for i, c in enumerate(hand_data):
            hand_features[i] = self._encode_card(c)
        
        # Get Top cards from solution
        solution = data.get('solution', {})
        top_data = solution.get('top', [])
        
        if len(top_data) != 3:
            return None
        
        # Find Top card indices in hand
        top_mask = np.zeros(self.max_cards, dtype=np.float32)
        valid_mask = np.zeros(self.max_cards, dtype=np.float32)
        valid_mask[:n_cards] = 1.0
        
        used = set()
        for tc in top_data:
            for i, hc in enumerate(hand_data):
                if i not in used:
                    if hc.get('rank') == tc.get('rank') and hc.get('suit') == tc.get('suit'):
                        top_mask[i] = 1.0
                        used.add(i)
                        break
        
        if top_mask.sum() != 3:
            return None
        
        return {
            'features': hand_features,
            'top_mask': top_mask,
            'valid_mask': valid_mask,
            'n_cards': n_cards
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = sample['features'].copy()
        top_mask = sample['top_mask'].copy()
        valid_mask = sample['valid_mask'].copy()
        n_cards = sample['n_cards']
        
        # Data augmentation: shuffle card order
        if self.augment:
            perm = np.random.permutation(n_cards)
            features[:n_cards] = features[perm]
            top_mask[:n_cards] = top_mask[perm]
        
        return {
            'features': torch.from_numpy(features),
            'top_mask': torch.from_numpy(top_mask),
            'valid_mask': torch.from_numpy(valid_mask),
            'n_cards': n_cards
        }


def compute_accuracy(scores: torch.Tensor, top_mask: torch.Tensor, valid_mask: torch.Tensor) -> float:
    """Compute Top-3 accuracy."""
    batch_size = scores.shape[0]
    correct = 0
    
    for b in range(batch_size):
        n_valid = int(valid_mask[b].sum().item())
        
        # Get top 3 predicted indices
        valid_scores = scores[b, :n_valid]
        pred_top = set(valid_scores.argsort(descending=True)[:3].tolist())
        
        # Get true top indices
        true_top = set(top_mask[b, :n_valid].nonzero(as_tuple=True)[0].tolist())
        
        # Accuracy: how many of the 3 are correct
        correct += len(pred_top & true_top) / 3.0
    
    return correct / batch_size


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_acc = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        features = batch['features'].to(device)
        top_mask = batch['top_mask'].to(device)
        valid_mask = batch['valid_mask'].to(device)
        
        optimizer.zero_grad()
        
        # Padding mask for attention (True = ignore)
        padding_mask = (valid_mask == 0)
        
        scores = model(features, mask=padding_mask)
        
        loss = criterion(scores, top_mask, valid_mask)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += compute_accuracy(scores.detach(), top_mask, valid_mask)
    
    return total_loss / len(dataloader), total_acc / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_acc = 0
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            top_mask = batch['top_mask'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            
            padding_mask = (valid_mask == 0)
            scores = model(features, mask=padding_mask)
            
            loss = criterion(scores, top_mask, valid_mask)
            
            total_loss += loss.item()
            total_acc += compute_accuracy(scores, top_mask, valid_mask)
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': total_acc / len(dataloader)
    }


def main():
    parser = argparse.ArgumentParser(description="Train Improved Policy Network")
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--model', type=str, choices=['ranking', 'transformer'], default='transformer')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--output', type=str, default='ai/models/policy_net_v2.pt')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no-augment', action='store_true')
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    print(f"Model type: {args.model}")
    
    # Load data
    dataset = CardRankingDataset(args.data, augment=not args.no_augment)
    
    # Split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Disable augmentation for validation
    val_dataset.dataset.augment = False
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Create model
    if args.model == 'ranking':
        model = CardRankingNetwork(hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    else:
        model = TransformerPolicyNetwork(
            d_model=args.hidden_dim,
            num_layers=args.num_layers,
            nhead=4,
            dim_feedforward=args.hidden_dim * 2
        )
    
    model = model.to(args.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = PairwiseRankingLoss(margin=1.0)
    
    # Training loop
    best_val_acc = 0
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, args.device)
        val_metrics = evaluate(model, val_loader, criterion, args.device)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'model_type': args.model,
                'val_accuracy': best_val_acc,
                'config': {
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers
                }
            }, args.output)
            print(f"  Saved best model (acc: {best_val_acc:.4f})")
    
    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
