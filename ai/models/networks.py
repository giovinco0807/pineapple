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


class PolicyNetwork(nn.Module):
    """
    Predicts a probability distribution over valid actions.

    Input:  490-dim state vector
    Output: MAX_ACTIONS-dim probability distribution (after masking)

    Architecture:
        490 → 512 (ReLU, Dropout) → 256 (ReLU, Dropout) → MAX_ACTIONS → Masked Softmax
    """

    def __init__(self, input_dim: int = STATE_DIM, hidden1: int = 512,
                 hidden2: int = 256, max_actions: int = MAX_ACTIONS,
                 dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, max_actions),
        )

    def forward(self, state: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state:      (batch, STATE_DIM) state vector
            valid_mask: (batch, MAX_ACTIONS) boolean mask for valid actions
        Returns:
            action_probs: (batch, MAX_ACTIONS) probability distribution
        """
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

    Input:  490-dim state vector
    Output: dict with royalty_ev, bust_prob, fl_prob, value
    """

    def __init__(self, input_dim: int = STATE_DIM, hidden1: int = 512,
                 hidden2: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
        )

        # Layer 1: Self-evaluation heads (trained in BC)
        self.royalty_head = nn.Linear(hidden2, 1)
        self.bust_head = nn.Linear(hidden2, 1)
        self.fl_head = nn.Linear(hidden2, 1)

        # Layer 2: Game outcome head (trained in Self-Play)
        self.value_head = nn.Linear(hidden2, 1)

    def forward(self, state: torch.Tensor) -> dict:
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
