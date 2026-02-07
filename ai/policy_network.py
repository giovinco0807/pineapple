"""
Policy Network for Fantasyland Solver

Predicts top-k placement candidates, then uses exhaustive search on those candidates only.
This dramatically reduces search space from 1M to ~15K combinations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from itertools import combinations
from pathlib import Path

# Card encoding dimensions
RANK_DIM = 13  # 2-A
SUIT_DIM = 4   # ♠♥♦♣
JOKER_DIM = 1  # is_joker flag
CARD_DIM = RANK_DIM + SUIT_DIM + JOKER_DIM  # 18

MAX_CARDS = 17  # Max FL cards


def encode_card(card) -> np.ndarray:
    """Encode a single card as a feature vector."""
    features = np.zeros(CARD_DIM, dtype=np.float32)
    
    if card.is_joker:
        features[RANK_DIM + SUIT_DIM] = 1.0  # Joker flag
    else:
        # Rank: 2=0, 3=1, ..., A=12
        rank_idx = card.rank_value - 2
        features[rank_idx] = 1.0
        
        # Suit: spades=0, hearts=1, diamonds=2, clubs=3
        suit_map = {'spades': 0, 'hearts': 1, 'diamonds': 2, 'clubs': 3}
        suit_idx = suit_map.get(card.suit, 0)
        features[RANK_DIM + suit_idx] = 1.0
    
    return features


def encode_hand(cards: List, pad_to: int = MAX_CARDS) -> np.ndarray:
    """Encode a hand of cards."""
    features = np.zeros((pad_to, CARD_DIM), dtype=np.float32)
    for i, card in enumerate(cards[:pad_to]):
        features[i] = encode_card(card)
    return features


def get_all_top_combinations(n_cards: int) -> List[Tuple[int, int, int]]:
    """Get all possible 3-card Top combinations as index tuples."""
    return list(combinations(range(n_cards), 3))


class TopPolicyNetwork(nn.Module):
    """
    Predicts which 3-card Top combinations are most promising.
    
    Input: Encoded hand (n_cards x 18)
    Output: Score for each Top combination
    """
    
    def __init__(
        self,
        max_cards: int = 17,
        hidden_dim: int = 256,
        num_layers: int = 3
    ):
        super().__init__()
        self.max_cards = max_cards
        
        # Input: flattened hand encoding
        input_dim = max_cards * CARD_DIM
        
        # Max possible Top combinations for 17 cards: C(17,3) = 680
        self.max_tops = len(get_all_top_combinations(max_cards))
        
        # Build network
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, self.max_tops))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, max_cards, CARD_DIM) or (batch, max_cards * CARD_DIM)
        
        Returns:
            (batch, max_tops) logits for each Top combination
        """
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        return self.network(x)
    
    def predict_top_k(
        self,
        hand_features: np.ndarray,
        k: int = 50,
        n_cards: int = 14
    ) -> List[Tuple[int, int, int]]:
        """
        Predict top-k best Top combinations for a hand.
        
        Args:
            hand_features: (max_cards, CARD_DIM) encoded hand
            k: number of candidates to return
            n_cards: actual number of cards in hand
        
        Returns:
            List of (i, j, k) index tuples for best Top combinations
        """
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(hand_features).unsqueeze(0)
            logits = self.forward(x).squeeze(0).numpy()
        
        # Get valid combinations for this hand size
        valid_tops = get_all_top_combinations(n_cards)
        
        # Score only valid combinations
        scores = logits[:len(valid_tops)]
        
        # Get top-k indices
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        return [valid_tops[i] for i in top_k_indices]


class MiddlePolicyNetwork(nn.Module):
    """
    Predicts which 5-card Middle combinations are most promising,
    given the selected Top.
    """
    
    def __init__(
        self,
        max_remaining: int = 14,  # After removing 3 for Top
        hidden_dim: int = 256,
        num_layers: int = 3
    ):
        super().__init__()
        self.max_remaining = max_remaining
        
        # Input: remaining cards + top cards
        input_dim = max_remaining * CARD_DIM + 3 * CARD_DIM
        
        # Max Middle combinations: C(14,5) = 2002
        self.max_middles = len(list(combinations(range(max_remaining), 5)))
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, self.max_middles))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        return self.network(x)


class PolicyGuidedSolver:
    """
    Solver that uses policy networks to guide search.
    
    Flow:
    1. Use TopPolicyNetwork to get top-k Top candidates
    2. For each Top, exhaustively search Middle/Bottom
    3. Return best placement found
    """
    
    def __init__(
        self,
        top_network: Optional[TopPolicyNetwork] = None,
        k_top: int = 50,
        device: str = 'cpu'
    ):
        self.top_network = top_network
        self.k_top = k_top
        self.device = device
        
        if top_network:
            self.top_network.to(device)
            self.top_network.eval()
    
    def solve(self, hand: List, use_policy: bool = True) -> dict:
        """
        Solve a Fantasyland hand.
        
        Args:
            hand: List of Card objects
            use_policy: If True, use policy network. If False, exhaustive.
        
        Returns:
            dict with 'top', 'middle', 'bottom', 'discards', 'score'
        """
        from fl_solver import evaluate_placement, is_valid_placement
        
        n = len(hand)
        
        if use_policy and self.top_network:
            # Get top-k candidates from policy network
            hand_features = encode_hand(hand, pad_to=17)
            top_candidates = self.top_network.predict_top_k(
                hand_features, k=self.k_top, n_cards=n
            )
        else:
            # All possible Tops
            top_candidates = list(combinations(range(n), 3))
        
        best_placement = None
        best_score = float('-inf')
        checked = 0
        
        for top_idx in top_candidates:
            top = [hand[i] for i in top_idx]
            remaining = [hand[i] for i in range(n) if i not in top_idx]
            
            # Exhaustive search for Middle/Bottom
            for mid_idx in combinations(range(len(remaining)), 5):
                middle = [remaining[i] for i in mid_idx]
                rest = [remaining[i] for i in range(len(remaining)) if i not in mid_idx]
                
                for bot_idx in combinations(range(len(rest)), 5):
                    bottom = [rest[i] for i in bot_idx]
                    discards = [rest[i] for i in range(len(rest)) if i not in bot_idx]
                    
                    checked += 1
                    
                    # Skip if joker in discards
                    if any(c.is_joker for c in discards):
                        continue
                    
                    placement = evaluate_placement(top, middle, bottom)
                    
                    if not placement.is_bust and placement.score > best_score:
                        best_score = placement.score
                        best_placement = {
                            'top': top,
                            'middle': middle,
                            'bottom': bottom,
                            'discards': discards,
                            'score': placement.score,
                            'royalties': placement.royalties,
                            'can_stay': placement.can_stay
                        }
        
        if best_placement:
            best_placement['checked'] = checked
        
        return best_placement
    
    def load_weights(self, path: str):
        """Load pretrained weights for policy networks."""
        checkpoint = torch.load(path, map_location=self.device)
        if self.top_network:
            self.top_network.load_state_dict(checkpoint['top_network'])


def create_training_labels(data_path: str, output_path: str):
    """
    Create training labels from solver data.
    
    For each hand in the data:
    - The correct Top is mapped to its combination index
    - This becomes the classification target
    """
    import json
    from game.card import Card
    
    labels = []
    
    with open(data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # Reconstruct hand
            hand = [Card(c['rank'], c['suit']) for c in data['hand']]
            n = len(hand)
            
            # Get correct Top
            top_cards = [Card(c['rank'], c['suit']) for c in data['top']]
            
            # Find Top indices in hand
            top_indices = []
            for tc in top_cards:
                for i, hc in enumerate(hand):
                    if hc.rank == tc.rank and hc.suit == tc.suit and i not in top_indices:
                        top_indices.append(i)
                        break
            
            if len(top_indices) != 3:
                continue  # Skip invalid
            
            top_indices = tuple(sorted(top_indices))
            
            # Find index in all combinations
            all_tops = list(combinations(range(n), 3))
            try:
                label = all_tops.index(top_indices)
            except ValueError:
                continue
            
            labels.append({
                'hand': [encode_card(c).tolist() for c in hand],
                'top_label': label,
                'n_cards': n
            })
    
    with open(output_path, 'w') as f:
        json.dump(labels, f)
    
    print(f"Created {len(labels)} training labels")
    return labels


if __name__ == "__main__":
    # Test the network
    print("Testing TopPolicyNetwork...")
    
    net = TopPolicyNetwork(max_cards=17, hidden_dim=256)
    print(f"Network parameters: {sum(p.numel() for p in net.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(4, 17, CARD_DIM)
    out = net(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # Test prediction
    hand_features = np.random.randn(17, CARD_DIM).astype(np.float32)
    top_k = net.predict_top_k(hand_features, k=10, n_cards=14)
    print(f"Top 10 predictions: {top_k[:5]}...")
