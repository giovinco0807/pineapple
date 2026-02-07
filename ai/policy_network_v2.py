"""
Improved Policy Network with Card Ranking Approach.

Instead of 364-class classification, we score each card individually
and select the top-3 scoring cards for Top placement.

This is much simpler and generalizes better.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from itertools import combinations
import math

# Card encoding
RANK_DIM = 13
SUIT_DIM = 4
JOKER_DIM = 1
CARD_DIM = RANK_DIM + SUIT_DIM + JOKER_DIM  # 18


class CardRankingNetwork(nn.Module):
    """
    Score each card for Top placement.
    
    Input: (batch, n_cards, CARD_DIM)
    Output: (batch, n_cards) - score for each card
    
    Select top-3 scoring cards as the Top.
    """
    
    def __init__(self, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        
        # Card embedding
        self.card_embed = nn.Linear(CARD_DIM, hidden_dim)
        
        # Self-attention to consider card relationships
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Feed-forward layers
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Output score for each card
        self.score_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, n_cards, CARD_DIM)
            mask: (batch, n_cards) - True for valid cards
        
        Returns:
            (batch, n_cards) - score for each card
        """
        # Embed cards
        h = self.card_embed(x)  # (batch, n_cards, hidden)
        
        # Self-attention
        h_attn, _ = self.attention(h, h, h, key_padding_mask=mask)
        h = h + h_attn  # Residual
        
        # MLP
        h = self.mlp(h)
        
        # Score each card
        scores = self.score_head(h).squeeze(-1)  # (batch, n_cards)
        
        return scores
    
    def predict_top_indices(self, x: torch.Tensor, n_cards: int = 14) -> List[int]:
        """
        Predict which 3 cards should go to Top.
        
        Returns:
            List of 3 indices for Top cards
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 2:
                x = x.unsqueeze(0)
            
            scores = self.forward(x).squeeze(0)
            
            # Get top 3 indices (only consider valid cards)
            valid_scores = scores[:n_cards]
            top_3 = valid_scores.argsort(descending=True)[:3].tolist()
            
            return sorted(top_3)


class TransformerPolicyNetwork(nn.Module):
    """
    Transformer-based policy network for Top selection.
    
    Uses self-attention to model card relationships.
    """
    
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_cards: int = 17
    ):
        super().__init__()
        self.d_model = d_model
        self.max_cards = max_cards
        
        # Card embedding
        self.card_embed = nn.Linear(CARD_DIM, d_model)
        
        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, max_cards, d_model) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output: score for each card
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, n_cards, CARD_DIM)
            mask: (batch, n_cards) - True for padded positions
        
        Returns:
            (batch, n_cards) - score for each card
        """
        batch_size, n_cards, _ = x.shape
        
        # Embed cards
        h = self.card_embed(x)  # (batch, n_cards, d_model)
        
        # Add positional encoding
        h = h + self.pos_embed[:, :n_cards, :]
        
        # Transformer
        h = self.transformer(h, src_key_padding_mask=mask)
        
        # Score each card
        scores = self.output_head(h).squeeze(-1)  # (batch, n_cards)
        
        return scores
    
    def predict_top_indices(self, x: torch.Tensor, n_cards: int = 14) -> List[int]:
        """Predict top 3 card indices."""
        self.eval()
        with torch.no_grad():
            if x.dim() == 2:
                x = x.unsqueeze(0)
            
            scores = self.forward(x).squeeze(0)
            valid_scores = scores[:n_cards]
            top_3 = valid_scores.argsort(descending=True)[:3].tolist()
            
            return sorted(top_3)


class PairwiseRankingLoss(nn.Module):
    """
    Pairwise ranking loss for card ranking.
    
    Cards in Top should score higher than cards not in Top.
    """
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        scores: torch.Tensor,
        top_mask: torch.Tensor,
        valid_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            scores: (batch, n_cards) - predicted scores
            top_mask: (batch, n_cards) - 1 for Top cards, 0 otherwise
            valid_mask: (batch, n_cards) - 1 for valid cards
        
        Returns:
            Scalar loss
        """
        batch_size, n_cards = scores.shape
        
        if valid_mask is None:
            valid_mask = torch.ones_like(scores)
        
        loss = 0
        count = 0
        
        for b in range(batch_size):
            # Get top and non-top indices
            top_idx = (top_mask[b] == 1).nonzero(as_tuple=True)[0]
            non_top_idx = ((top_mask[b] == 0) & (valid_mask[b] == 1)).nonzero(as_tuple=True)[0]
            
            if len(top_idx) == 0 or len(non_top_idx) == 0:
                continue
            
            # For each top card, it should score higher than all non-top cards
            for t_idx in top_idx:
                for nt_idx in non_top_idx:
                    # Hinge loss: max(0, margin - (score_top - score_non_top))
                    diff = scores[b, t_idx] - scores[b, nt_idx]
                    loss += F.relu(self.margin - diff)
                    count += 1
        
        if count > 0:
            loss = loss / count
        
        return loss


def augment_hand(hand_features: np.ndarray, n_cards: int) -> np.ndarray:
    """
    Data augmentation: randomly permute card order.
    
    The model should be invariant to card order.
    """
    perm = np.random.permutation(n_cards)
    augmented = hand_features.copy()
    augmented[:n_cards] = hand_features[perm]
    
    return augmented, perm


if __name__ == "__main__":
    print("Testing CardRankingNetwork...")
    
    model = CardRankingNetwork(hidden_dim=128)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test
    x = torch.randn(4, 14, CARD_DIM)
    scores = model(x)
    print(f"Input: {x.shape}, Output: {scores.shape}")
    
    top_indices = model.predict_top_indices(x[0], n_cards=14)
    print(f"Top indices: {top_indices}")
    
    print("\nTesting TransformerPolicyNetwork...")
    
    model2 = TransformerPolicyNetwork(d_model=128, num_layers=3)
    print(f"Parameters: {sum(p.numel() for p in model2.parameters()):,}")
    
    scores2 = model2(x)
    print(f"Input: {x.shape}, Output: {scores2.shape}")
