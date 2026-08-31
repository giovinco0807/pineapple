"""Set/listwise action-value reranker for candidate groups.

Unlike ``ActionValueReranker``, this model receives every legal candidate for a
decision at once.  Candidate embeddings can therefore compare against the local
set before producing a per-candidate score.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.engine.encoding import STATE_DIM
from ai.models.networks import _adapt_state


class SetResidualBlock(nn.Module):
    def __init__(self, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class ActionValueSetReranker(nn.Module):
    """Scores all candidates in a decision group jointly."""

    def __init__(
        self,
        input_dim: int = STATE_DIM,
        hidden: int = 384,
        n_blocks: int = 2,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        base_score_residual: bool = False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.n_blocks = int(n_blocks)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.dropout = float(dropout)
        self.base_score_residual = bool(base_score_residual)
        self.score_mean = 0.0
        self.score_std = 1.0

        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList([SetResidualBlock(hidden, dropout) for _ in range(n_blocks)])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 3),
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        if self.base_score_residual:
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def forward(self, state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return normalized scores with shape ``[batch, candidates]``."""
        if state.ndim != 3:
            raise ValueError("state must have shape [batch, candidates, features]")
        if mask.ndim != 2:
            raise ValueError("mask must have shape [batch, candidates]")
        batch, candidates, _ = state.shape
        base_score = state[..., -1] if self.base_score_residual else None
        state = _adapt_state(state.reshape(batch * candidates, state.shape[-1]), self.input_dim)
        x = self.input_proj(state).reshape(batch, candidates, self.hidden)
        for block in self.blocks:
            x = block(x)

        key_padding_mask = ~mask.bool()
        x = self.set_encoder(x, src_key_padding_mask=key_padding_mask)
        mask_f = mask.to(dtype=x.dtype).unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        mean_context = (x * mask_f).sum(dim=1) / denom
        masked_x = x.masked_fill(~mask.bool().unsqueeze(-1), -1e9)
        max_context = masked_x.max(dim=1).values
        max_context = max_context.masked_fill(max_context < -1e8, 0.0)
        context = torch.cat([mean_context, max_context], dim=-1).unsqueeze(1).expand(-1, candidates, -1)
        logits = self.head(torch.cat([x, context], dim=-1)).squeeze(-1)
        if base_score is not None:
            logits = logits + base_score
        return logits.masked_fill(~mask.bool(), -1e9)

    def predict_scores(self, state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        score = self.forward(state, mask)
        return score * float(self.score_std) + float(self.score_mean)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        map_location: str | torch.device | None = None,
    ) -> "ActionValueSetReranker":
        ckpt: Any = torch.load(path, map_location=map_location or "cpu", weights_only=False)
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            raise ValueError(f"Unsupported set reranker checkpoint: {path}")
        config = ckpt.get("model_config", {})
        model = cls(
            input_dim=int(config.get("input_dim", ckpt.get("input_dim", STATE_DIM))),
            hidden=int(config.get("hidden", 384)),
            n_blocks=int(config.get("n_blocks", 2)),
            n_layers=int(config.get("n_layers", 2)),
            n_heads=int(config.get("n_heads", 8)),
            dropout=float(config.get("dropout", 0.1)),
            base_score_residual=bool(config.get("base_score_residual", False)),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        norm = ckpt.get("normalization", {})
        model.score_mean = float(norm.get("score_mean", ckpt.get("score_mean", 0.0)))
        model.score_std = float(norm.get("score_std", ckpt.get("score_std", 1.0)))
        return model
