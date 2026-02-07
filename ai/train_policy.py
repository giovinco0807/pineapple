"""
Training script for Policy Network.

Usage:
    python train_policy.py --data ai/data/fl_joker0_v2.jsonl --epochs 50 --output ai/models/policy_net.pt
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
from itertools import combinations

import sys
sys.path.insert(0, str(Path(__file__).parent))

from policy_network import (
    TopPolicyNetwork,
    encode_card,
    encode_hand,
    get_all_top_combinations,
    CARD_DIM
)


class TopPolicyDataset(Dataset):
    """Dataset for training Top prediction network."""
    
    def __init__(self, data_path: str, max_cards: int = 17):
        self.max_cards = max_cards
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
    
    def _process_sample(self, data: dict) -> dict:
        """Process a single sample from JSONL."""
        # Get hand
        hand_data = data.get('hand', [])
        n_cards = len(hand_data)
        
        if n_cards < 13 or n_cards > 17:
            return None
        
        # Encode hand
        hand_features = np.zeros((self.max_cards, CARD_DIM), dtype=np.float32)
        for i, c in enumerate(hand_data):
            # Create a simple Card-like object
            class TempCard:
                def __init__(self, rank, suit):
                    self.rank = rank
                    self.suit = suit
                    self.is_joker = rank == 'Joker' or suit == 'joker'
                    # Rank value
                    rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                               '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
                    self.rank_value = rank_map.get(rank, 0)
            
            card = TempCard(c.get('rank', ''), c.get('suit', ''))
            hand_features[i] = encode_card(card)
        
        # Get correct Top indices from solution
        solution = data.get('solution', {})
        top_data = solution.get('top', [])
        if len(top_data) != 3:
            return None
        
        # Find Top card indices in hand
        top_indices = []
        used = set()
        for tc in top_data:
            for i, hc in enumerate(hand_data):
                if i not in used and hc.get('rank') == tc.get('rank') and hc.get('suit') == tc.get('suit'):
                    top_indices.append(i)
                    used.add(i)
                    break
        
        if len(top_indices) != 3:
            return None
        
        top_indices = tuple(sorted(top_indices))
        
        # Find index in all combinations
        all_tops = get_all_top_combinations(n_cards)
        try:
            label = all_tops.index(top_indices)
        except ValueError:
            return None
        
        return {
            'features': hand_features,
            'label': label,
            'n_cards': n_cards
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'features': torch.from_numpy(sample['features']),
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'n_cards': sample['n_cards']
        }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        features = batch['features'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        logits = model(features)
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Accuracy
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device, k=10):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    top_k_correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(features)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            # Top-1 accuracy
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            
            # Top-k accuracy
            top_k_preds = logits.topk(k, dim=1).indices
            for i, label in enumerate(labels):
                if label in top_k_preds[i]:
                    top_k_correct += 1
            
            total += labels.size(0)
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / total,
        f'top_{k}_accuracy': top_k_correct / total
    }


def main():
    parser = argparse.ArgumentParser(description="Train Top Policy Network")
    parser.add_argument('--data', type=str, required=True, help='Path to training data (JSONL)')
    parser.add_argument('--val-data', type=str, help='Path to validation data (optional)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--output', type=str, default='ai/models/policy_net.pt')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    # Load data
    dataset = TopPolicyDataset(args.data)
    
    # Split into train/val if no val data provided
    if args.val_data:
        val_dataset = TopPolicyDataset(args.val_data)
        train_dataset = dataset
    else:
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Create model
    model = TopPolicyNetwork(
        max_cards=17,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, args.device)
        val_metrics = evaluate(model, val_loader, criterion, args.device, k=10)
        
        scheduler.step(val_metrics['loss'])
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, Top-10: {val_metrics['top_10_accuracy']:.4f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            
            # Ensure output directory exists
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
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
