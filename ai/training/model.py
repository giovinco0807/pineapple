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


class TurnValueModel(nn.Module):
    """
    Value network for Turn states (T1-T4).
    Input: Partially filled board (up to 14 cards, padded to 14) + row indices.
    Output: EV scalar.
    """
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
        
        # Row embeddings: 0=top, 1=mid, 2=bot
        self.row_embed = nn.Embedding(4, hidden_dim, padding_idx=3)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # CLS token for pooling
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, board, rows):
        """
        Args:
            board: [batch, 14, 19] card features
            rows: [batch, 14] row labels (0=top, 1=mid, 2=bot, -1=pad)
        Returns:
            ev: [batch, 1] scalar Expected Value
        """
        batch_size, max_len, _ = board.shape
        
        # handle padding in rows (-1 -> 3 for embedding)
        rows_embed_idx = torch.where(rows == -1, torch.tensor(3, device=rows.device), rows)
        
        x = self.card_embed(board) + self.row_embed(rows_embed_idx)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Padding mask
        mask = torch.cat((
            torch.zeros(batch_size, 1, dtype=torch.bool, device=rows.device),
            rows == -1
        ), dim=1)
        
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Use CLS token for value prediction
        cls_out = x[:, 0, :]
        ev = self.value_head(cls_out)
        
        return ev

