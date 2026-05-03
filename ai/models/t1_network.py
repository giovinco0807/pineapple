import torch
import torch.nn as nn
import numpy as np

CARD_DIM_BASE = 18  # 13 ranks + 4 suits + 1 joker flag
BOARD_ROW_DIM = 4   # 0: Hand, 1: Top, 2: Mid, 3: Bot
CARD_DIM = CARD_DIM_BASE + BOARD_ROW_DIM  # 22
NUM_CLASSES = 4     # Top=0, Mid=1, Bot=2, Discard=3
MAX_CARDS = 8       # 5 on board + 3 in hand
SUITS = ['s', 'h', 'd', 'c']

def encode_card_str(card_str: str, board_row: int) -> np.ndarray:
    """Encode a card string like '9s' or 'X1' into base features + board row."""
    features = np.zeros(CARD_DIM, dtype=np.float32)
    if card_str in ('JK', 'X1', 'X2'):
        features[17] = 1.0
    else:
        rank = card_str[0]
        suit_char = card_str[1]
        rank_map = {'2':0,'3':1,'4':2,'5':3,'6':4,'7':5,
                   '8':6,'9':7,'T':8,'J':9,'Q':10,'K':11,'A':12}
        r = rank_map.get(rank, 0)
        features[r] = 1.0
        suit_map = {'s':0,'h':1,'d':2,'c':3}
        s = suit_map.get(suit_char, 0)
        features[13 + s] = 1.0
    features[18 + board_row] = 1.0
    return features

class T1PlacementNet(nn.Module):
    """Transformer for T1 card placement."""

    def __init__(self, d_model=128, nhead=4, num_layers=4, dim_ff=256, dropout=0.2):
        super().__init__()
        self.card_embed = nn.Sequential(
            nn.Linear(CARD_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, MAX_CARDS, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.row_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, NUM_CLASSES),
        )
        self.ev_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, cards):
        # cards: (B, 8, CARD_DIM)
        x = self.card_embed(cards) + self.pos_embed[:, :cards.size(1)]
        x = self.encoder(x)
        # only predict for the hand cards (last 3 cards)
        hand_tokens = x[:, -3:, :]
        row_logits = self.row_head(hand_tokens)
        ev_pred = self.ev_head(x.mean(dim=1))
        return row_logits, ev_pred
