"""
OFC Pineapple - Neural Network Models

PolicyNetwork: Predicts action probability distribution
ValueNetwork: Predicts royalty EV, bust/FL probabilities, and game value
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.engine.encoding import STATE_DIM
from ai.engine.action_space import MAX_ACTIONS


def _adapt_state(state: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Adapt state tensor to target dimension (handles 522<->520 compat)."""
    src_dim = state.shape[-1]
    if src_dim == target_dim:
        return state
    if src_dim == 522 and target_dim == 520:
        # Remove is_fl/opp_is_fl at indices 488,489 (meta[2:4])
        return torch.cat([state[..., :488], state[..., 490:]], dim=-1)
    if src_dim > target_dim:
        return state[..., :target_dim]
    # src_dim < target_dim: pad with zeros
    pad = torch.zeros(*state.shape[:-1], target_dim - src_dim,
                       device=state.device, dtype=state.dtype)
    return torch.cat([state, pad], dim=-1)


class PolicyNetwork(nn.Module):
    """
    Predicts a probability distribution over valid actions.

    Input:  520-dim state vector
    Output: MAX_ACTIONS-dim probability distribution (after masking)

    Architecture:
        520 → 1024 (ReLU, Dropout) → 512 (ReLU, Dropout) → 256 (ReLU, Dropout) → MAX_ACTIONS
    """

    def __init__(self, input_dim: int = STATE_DIM,
                 max_actions: int = MAX_ACTIONS,
                 dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, max_actions),
        )

    def forward(self, state: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state:      (batch, STATE_DIM) state vector
            valid_mask: (batch, MAX_ACTIONS) boolean mask for valid actions
        Returns:
            action_probs: (batch, MAX_ACTIONS) probability distribution
        """
        state = _adapt_state(state, self.net[0].in_features)
        logits = self.net(state)
        logits = logits.masked_fill(~valid_mask, float('-inf'))
        return F.softmax(logits, dim=-1)

    def select_action(self, state: torch.Tensor, valid_mask: torch.Tensor,
                      temperature: float = 1.0) -> torch.Tensor:
        """
        Select an action index.
        temperature=0: greedy (best action)
        temperature>0: stochastic sampling
        """
        with torch.no_grad():
            state = _adapt_state(state, self.net[0].in_features)
            logits = self.net(state)
            logits = logits.masked_fill(~valid_mask, float('-inf'))

            if temperature == 0:
                return torch.argmax(logits, dim=-1)

            scaled = logits / max(temperature, 1e-8)
            probs = F.softmax(scaled, dim=-1)
            return torch.multinomial(probs, 1).squeeze(-1)


class ValueNetwork(nn.Module):
    """
    Evaluates board position quality.

    Layer 1 heads (BC stage): royalty_ev, bust_prob, fl_prob
    Layer 2 head (Self-Play stage): value

    Input:  520-dim state vector
    Output: dict with royalty_ev, bust_prob, fl_prob, value
    """

    def __init__(self, input_dim: int = STATE_DIM):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # Layer 1: Self-evaluation heads (trained in BC)
        self.royalty_head = nn.Linear(256, 1)
        self.bust_head = nn.Linear(256, 1)
        self.fl_head = nn.Linear(256, 1)

        # Layer 2: Game outcome head (trained in Self-Play)
        self.value_head = nn.Linear(256, 1)

    def forward(self, state: torch.Tensor) -> dict:
        state = _adapt_state(state, self.shared[0].in_features)
        x = self.shared(state)
        return {
            "royalty_ev": self.royalty_head(x),
            "bust_prob": torch.sigmoid(self.bust_head(x)),
            "fl_prob": torch.sigmoid(self.fl_head(x)),
            "value": self.value_head(x),
        }

    def freeze_shared(self):
        """Freeze shared layers (for fine-tuning heads only)."""
        for param in self.shared.parameters():
            param.requires_grad = False

    def unfreeze_shared(self):
        """Unfreeze shared layers."""
        for param in self.shared.parameters():
            param.requires_grad = True


# =====================================================================
# V2 Architecture: Residual Connections + LayerNorm
# =====================================================================

class ResidualBlock(nn.Module):
    """Pre-norm residual block: LayerNorm → FC → ReLU → Dropout → FC → Dropout + Skip."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class PolicyNetworkV2(nn.Module):
    """
    Policy network with residual connections and layer normalization.

    Architecture:
        520 → 1024 → ReLU → 512 → [ResBlock × 3 @ 512] → LN → 256 → ReLU → MAX_ACTIONS
    """

    def __init__(self, input_dim: int = STATE_DIM,
                 max_actions: int = MAX_ACTIONS,
                 hidden: int = 512, n_blocks: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, hidden),
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden, dropout) for _ in range(n_blocks)
        ])
        self.output = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Linear(256, max_actions),
        )

    def forward(self, state: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        state = _adapt_state(state, self.input_proj[0].in_features)
        x = self.input_proj(state)
        for block in self.blocks:
            x = block(x)
        logits = self.output(x)
        logits = logits.masked_fill(~valid_mask, float('-inf'))
        return F.softmax(logits, dim=-1)

    def select_action(self, state: torch.Tensor, valid_mask: torch.Tensor,
                      temperature: float = 1.0) -> torch.Tensor:
        with torch.no_grad():
            state = _adapt_state(state, self.input_proj[0].in_features)
            x = self.input_proj(state)
            for block in self.blocks:
                x = block(x)
            logits = self.output(x)
            logits = logits.masked_fill(~valid_mask, float('-inf'))

            if temperature == 0:
                return torch.argmax(logits, dim=-1)

            scaled = logits / max(temperature, 1e-8)
            probs = F.softmax(scaled, dim=-1)
            return torch.multinomial(probs, 1).squeeze(-1)


class ValueNetworkV2(nn.Module):
    """
    Value network with residual connections and layer normalization.

    Architecture:
        520 → 1024 → ReLU → 512 → [ResBlock × 3 @ 512] → LN → 256 → [4 heads]
    """

    def __init__(self, input_dim: int = STATE_DIM,
                 hidden: int = 512, n_blocks: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, hidden),
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden, dropout) for _ in range(n_blocks)
        ])
        self.trunk_out = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 256),
            nn.ReLU(),
        )

        # Layer 1: Self-evaluation heads (trained in BC)
        self.royalty_head = nn.Linear(256, 1)
        self.bust_head = nn.Linear(256, 1)
        self.fl_head = nn.Linear(256, 1)

        # Layer 2: Game outcome head (trained in Self-Play)
        self.value_head = nn.Linear(256, 1)

    def forward(self, state: torch.Tensor) -> dict:
        state = _adapt_state(state, self.input_proj[0].in_features)
        x = self.input_proj(state)
        for block in self.blocks:
            x = block(x)
        x = self.trunk_out(x)
        return {
            "royalty_ev": self.royalty_head(x),
            "bust_prob": torch.sigmoid(self.bust_head(x)),
            "fl_prob": torch.sigmoid(self.fl_head(x)),
            "value": self.value_head(x),
        }

    def freeze_shared(self):
        for param in self.input_proj.parameters():
            param.requires_grad = False
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False

    def unfreeze_shared(self):
        for param in self.input_proj.parameters():
            param.requires_grad = True
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = True


# =====================================================================
# V3 Architecture: Turn-Conditioned Heads
# =====================================================================

class ValueNetworkV3(nn.Module):
    """
    Value network with turn-conditioned bust/FL heads.

    Same shared layers as V1, but bust/FL heads receive a learned
    turn embedding concatenated with the shared features.
    This lets the heads specialize per-turn (e.g., T0 bust prediction
    differs fundamentally from T4 bust prediction).

    Input:  520-dim state vector + turn index (0-4)
    Output: dict with royalty_ev, bust_prob, fl_prob, value
    """

    def __init__(self, input_dim: int = STATE_DIM, n_turns: int = 5,
                 turn_embed_dim: int = 16):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # Turn embedding
        self.turn_embed = nn.Embedding(n_turns, turn_embed_dim)

        # Turn-conditioned heads: 256 + turn_embed_dim
        head_input = 256 + turn_embed_dim
        self.bust_head = nn.Sequential(
            nn.Linear(head_input, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.fl_head = nn.Sequential(
            nn.Linear(head_input, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Non-conditioned heads
        self.royalty_head = nn.Linear(256, 1)
        self.value_head = nn.Linear(256, 1)

    def forward(self, state: torch.Tensor, turn: torch.Tensor = None) -> dict:
        state = _adapt_state(state, self.shared[0].in_features)
        x = self.shared(state)
        result = {
            "royalty_ev": self.royalty_head(x),
            "value": self.value_head(x),
        }

        if turn is not None:
            t_emb = self.turn_embed(turn)  # (batch, turn_embed_dim)
            xt = torch.cat([x, t_emb], dim=-1)
        else:
            # Fallback: zero embedding (for inference without turn info)
            zero_emb = torch.zeros(x.size(0), self.turn_embed.embedding_dim,
                                   device=x.device)
            xt = torch.cat([x, zero_emb], dim=-1)

        result["bust_prob"] = torch.sigmoid(self.bust_head(xt))
        result["fl_prob"] = torch.sigmoid(self.fl_head(xt))
        return result

    def load_v1_weights(self, v1_state_dict: dict):
        """Initialize shared layers from a V1 checkpoint."""
        own = self.state_dict()
        loaded = 0
        for k, v in v1_state_dict.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                loaded += 1
        self.load_state_dict(own)
        return loaded

    def freeze_shared(self):
        for param in self.shared.parameters():
            param.requires_grad = False

    def unfreeze_shared(self):
        for param in self.shared.parameters():
            param.requires_grad = True
