"""
FL Supervised Learning Training

Trains a neural network to predict optimal FL placement from 14 cards.

Usage:
    python train_supervised.py --input ai/data/fl_training_14.jsonl --epochs 100
"""
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. Install with: pip install torch")


# Card encoding constants
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['s', 'h', 'd', 'c']  # spades, hearts, diamonds, clubs
NUM_CARDS = 52 + 2  # 52 normal + 2 jokers


def card_to_index(card_dict: Dict[str, str]) -> int:
    """Convert card dict to index (0-53)."""
    if card_dict.get('joker'):
        return 52 if card_dict.get('id', 1) == 1 else 53
    
    rank_idx = RANKS.index(card_dict['r'])
    suit_idx = SUITS.index(card_dict['s'])
    return suit_idx * 13 + rank_idx


def encode_hand(hand: List[Dict]) -> np.ndarray:
    """
    Encode 14 cards as a binary vector of shape (54,).
    1 at index i means card i is in hand.
    """
    vec = np.zeros(NUM_CARDS, dtype=np.float32)
    for card in hand:
        idx = card_to_index(card)
        vec[idx] = 1.0
    return vec


def encode_solution(solution: Dict) -> np.ndarray:
    """
    Encode solution as placement indices.
    Returns vector of shape (13,) with row assignments (0=top, 1=mid, 2=bot).
    """
    placement = np.zeros(13, dtype=np.int64)
    
    # Map each card to its placement
    card_to_row = {}
    for card in solution['top']:
        card_to_row[card_to_index(card)] = 0
    for card in solution['middle']:
        card_to_row[card_to_index(card)] = 1
    for card in solution['bottom']:
        card_to_row[card_to_index(card)] = 2
    
    # Sort by card index to get consistent ordering
    placed_indices = sorted(card_to_row.keys())
    for i, idx in enumerate(placed_indices):
        if i < 13:
            placement[i] = card_to_row[idx]
    
    return placement


class FLDataset(Dataset):
    """Dataset for FL training data."""
    
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Input: encoded hand
        hand_vec = encode_hand(item['hand'])
        
        # Target: reward (for regression) or placement (for classification)
        reward = item['reward']
        
        return torch.FloatTensor(hand_vec), torch.FloatTensor([reward])


class FLRewardPredictor(nn.Module):
    """
    Neural network to predict expected reward from a hand.
    
    Input: 54-dim binary vector (cards in hand)
    Output: Predicted reward (royalties + FL stay bonus)
    """
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(NUM_CARDS, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        return self.network(x)


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL data file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                pass
    return data


def train_model(
    data: List[Dict[str, Any]],
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> nn.Module:
    """Train the reward predictor model."""
    
    print(f"Training on device: {device}")
    print(f"Data size: {len(data)}")
    
    # Create dataset
    dataset = FLDataset(data)
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model
    model = FLRewardPredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for hands, rewards in train_loader:
            hands = hands.to(device)
            rewards = rewards.to(device)
            
            optimizer.zero_grad()
            outputs = model(hands)
            loss = criterion(outputs, rewards)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for hands, rewards in val_loader:
                hands = hands.to(device)
                rewards = rewards.to(device)
                
                outputs = model(hands)
                loss = criterion(outputs, rewards)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model
            torch.save(model.state_dict(), 'ai/models/fl_reward_predictor.pt')
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
    
    print()
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: ai/models/fl_reward_predictor.pt")
    
    return model


def main():
    if not TORCH_AVAILABLE:
        print("PyTorch is required. Install with: pip install torch")
        return
    
    parser = argparse.ArgumentParser(description='Train FL supervised model')
    parser.add_argument('--input', type=str, default='ai/data/fl_training_14.jsonl',
                        help='Input JSONL file')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: File not found: {args.input}")
        return
    
    # Create models directory
    Path('ai/models').mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_data(args.input)
    if not data:
        print("No data found!")
        return
    
    print(f"Loaded {len(data)} samples")
    
    # Train model
    model = train_model(
        data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )


if __name__ == '__main__':
    main()
