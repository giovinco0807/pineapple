"""
Transformer model for Fantasyland card placement.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=17):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class FantasylandModel(nn.Module):
    def __init__(
        self, 
        card_dim=19, 
        hidden_dim=128, 
        n_heads=4, 
        n_layers=4,
        dropout=0.1
    ):
        super().__init__()
        
        self.card_embed = nn.Sequential(
            nn.Linear(card_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4)  # top/middle/bottom/discard
        )
    
    def forward(self, hand, n_cards=None):
        """
        Args:
            hand: [batch, 17, 19] カード特徴量
            n_cards: [batch] 実際のカード枚数
        
        Returns:
            logits: [batch, 17, 4] 各カードの配置先予測
        """
        batch_size, max_len, _ = hand.shape
        
        # パディングマスク作成
        if n_cards is not None:
            mask = torch.arange(max_len, device=hand.device).expand(batch_size, -1) >= n_cards.unsqueeze(1)
        else:
            mask = None
        
        x = self.card_embed(hand)
        x = self.pos_encoding(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        logits = self.classifier(x)
        
        return logits
    
    def predict(self, hand, n_cards=None):
        """
        推論用: 配置先を予測
        
        Returns:
            placements: [batch, 17] 各カードの配置先 (0=top, 1=mid, 2=bot, 3=discard)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(hand, n_cards)
            return logits.argmax(dim=-1)


class FantasylandModelLarge(FantasylandModel):
    """より大きなモデル"""
    def __init__(self):
        super().__init__(
            card_dim=19,
            hidden_dim=256,
            n_heads=8,
            n_layers=6,
            dropout=0.1
        )
