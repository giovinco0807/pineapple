"""
OFC Pineapple - Monte Carlo Tree Search Engine

IS-MCTS with Progressive Widening for imperfect information OFC.
Uses BC-trained PolicyNet for priors and ValueNet for leaf evaluation.
"""
import sys
import math
import random
import copy
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.encoding import Board, Observation, encode_state, ALL_CARDS
from ai.engine.action_space import (
    get_initial_actions, get_turn_actions, Action, create_action_mask, MAX_ACTIONS
)
from ai.engine.game_engine import GameEngine, Hand


# ---------------------------------------------------------------------------
# MCTSNode
# ---------------------------------------------------------------------------
class MCTSNode:
    """A single node in the MCTS tree."""

    __slots__ = [
        "state", "parent", "action_idx", "prior",
        "children", "visit_count", "value_sum",
        "valid_actions", "is_expanded",
    ]

    def __init__(
        self,
        state: Optional[Observation] = None,
        parent: Optional["MCTSNode"] = None,
        action_idx: int = -1,
        prior: float = 0.0,
    ):
        self.state = state
        self.parent = parent
        self.action_idx = action_idx
        self.prior = prior
        self.children: Dict[int, "MCTSNode"] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.valid_actions: List[Action] = []
        self.is_expanded = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits: int, c_puct: float = 1.5) -> float:
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + exploration


# ---------------------------------------------------------------------------
# MCTS Config
# ---------------------------------------------------------------------------
@dataclass
class MCTSConfig:
    num_simulations: int = 200
    c_puct: float = 1.5
    temperature: float = 1.0       # Action selection temperature
    progressive_widening_c: float = 2.0
    progressive_widening_alpha: float = 0.5
    max_children: int = 50         # Cap for progressive widening
    dirichlet_alpha: float = 0.3   # Root exploration noise
    dirichlet_frac: float = 0.25   # Fraction of noise to mix in


# ---------------------------------------------------------------------------
# MCTS Search
# ---------------------------------------------------------------------------
class MCTS:
    """
    IS-MCTS with Progressive Widening for OFC Pineapple.

    Each simulation:
    1. Determinize: sample unseen cards to create a "possible world"
    2. Select: follow UCB1 down the tree
    3. Expand: add child nodes using PolicyNet priors
    4. Evaluate: use ValueNet on leaf
    5. Backpropagate: update visit counts and values
    """

    def __init__(
        self,
        policy_net: torch.nn.Module,
        value_net: torch.nn.Module,
        config: MCTSConfig = MCTSConfig(),
        device: str = "cpu",
    ):
        self.policy_net = policy_net
        self.value_net = value_net
        self.config = config
        self.device = device
        self.policy_net.eval()
        self.value_net.eval()

    def search(
        self,
        obs: Observation,
        board_state: dict,
    ) -> Tuple[int, Dict[int, float], List[Action]]:
        """
        Run MCTS from the given observation.

        Args:
            obs: current player's observation
            board_state: dict with board/deck info for simulation

        Returns:
            best_action_idx: index into valid_actions
            action_probs: {action_idx: probability} from visit counts
            valid_actions: list of valid Action objects
        """
        # Get valid actions
        if obs.turn == 0:
            valid_actions = get_initial_actions(obs.dealt_cards, obs.board_self)
        else:
            valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)

        if not valid_actions:
            return 0, {0: 1.0}, []

        if len(valid_actions) == 1:
            return 0, {0: 1.0}, valid_actions

        # Create root
        root = MCTSNode(state=obs)
        root.valid_actions = valid_actions

        # Expand root with policy priors
        priors = self._get_priors(obs, valid_actions)

        # Add Dirichlet noise at root for exploration
        noise = np.random.dirichlet(
            [self.config.dirichlet_alpha] * len(valid_actions)
        )
        frac = self.config.dirichlet_frac
        for i in range(len(valid_actions)):
            priors[i] = (1 - frac) * priors[i] + frac * noise[i]

        self._expand_root(root, priors)

        # Run simulations
        for _ in range(self.config.num_simulations):
            node = self._select(root)
            value = self._evaluate(node, obs)
            self._backpropagate(node, value)

        # Extract action probabilities from visit counts
        action_probs = self._get_action_probs(root)
        best_action_idx = self._select_action(action_probs)

        return best_action_idx, action_probs, valid_actions

    def _get_priors(self, obs: Observation, valid_actions: List[Action]) -> np.ndarray:
        """Get policy network priors for valid actions."""
        state_vec = encode_state(obs)
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)

        mask = create_action_mask(valid_actions)
        mask_tensor = torch.BoolTensor(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = self.policy_net(state_tensor, mask_tensor).squeeze(0).cpu().numpy()

        # Extract priors for valid actions only
        priors = probs[:len(valid_actions)]
        total = priors.sum()
        if total > 0:
            priors = priors / total
        else:
            priors = np.ones(len(valid_actions)) / len(valid_actions)

        return priors

    def _expand_root(self, root: MCTSNode, priors: np.ndarray):
        """Expand root node with all valid actions."""
        for i, action in enumerate(root.valid_actions):
            child = MCTSNode(
                parent=root,
                action_idx=i,
                prior=priors[i],
            )
            child.valid_actions = []
            root.children[i] = child
        root.is_expanded = True

    def _select(self, root: MCTSNode) -> MCTSNode:
        """Select leaf node using UCB1."""
        node = root
        while node.is_expanded and node.children:
            # Progressive widening: limit children
            k = min(
                len(node.children),
                max(1, int(math.ceil(
                    self.config.progressive_widening_c *
                    (node.visit_count ** self.config.progressive_widening_alpha)
                )))
            )
            k = min(k, self.config.max_children)

            # Select among top-k children by prior (sorted by visits then UCB)
            candidates = list(node.children.values())[:k] if k < len(node.children) else list(node.children.values())

            node = max(
                candidates,
                key=lambda c: c.ucb_score(node.visit_count, self.config.c_puct)
            )
        return node

    def _evaluate(self, node: MCTSNode, root_obs: Observation) -> float:
        """
        Evaluate a leaf node using the value network.

        Uses a composite value from BC-trained heads (royalty_ev, bust_prob,
        fl_prob) rather than the untrained 'value' head. The value head
        only becomes meaningful after self-play training updates it.
        """
        # Build observation for this node by simulating the action path
        obs = self._build_node_observation(node, root_obs)
        if obs is None:
            return 0.0

        state_vec = encode_state(obs)
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.value_net(state_tensor)

        # Composite value from BC-learned heads:
        #   royalty_ev: expected total royalties (directly additive to score)
        #   bust_prob: probability of busting (penalty ≈ -6 scoop - ~5 opp royalty)
        #   fl_prob: probability of FL entry (bonus ≈ +8 expected advantage)
        royalty = pred["royalty_ev"].item()
        bust = pred["bust_prob"].item()
        fl = pred["fl_prob"].item()
        value = royalty - bust * 11.0 + fl * 8.0

        # Normalize to [-1, 1] range (typical game values ~[-20, 30])
        return max(-1.0, min(1.0, value / 20.0))

    def _build_node_observation(
        self, node: MCTSNode, root_obs: Observation
    ) -> Optional[Observation]:
        """
        Build the observation at a node by applying actions from root.
        Walks the tree path and applies each node's action using its
        parent's valid_actions list.
        """
        # Walk up to root to collect path of nodes
        node_path = []
        current = node
        while current.parent is not None:
            node_path.append(current)
            current = current.parent
        node_path.reverse()

        # Start from root observation, apply actions along path
        board = Board(
            top=list(root_obs.board_self.top),
            middle=list(root_obs.board_self.middle),
            bottom=list(root_obs.board_self.bottom),
        )
        dealt = list(root_obs.dealt_cards)

        # Walk down: each child_node's action_idx indexes into its parent's valid_actions
        parent = current  # root node
        for child_node in node_path:
            valid = parent.valid_actions if parent.valid_actions else []
            idx = child_node.action_idx
            if idx < len(valid):
                action = valid[idx]
                for card, pos in action.placements:
                    if pos == "top" and len(board.top) < 3:
                        board.top.append(card)
                    elif pos == "middle" and len(board.middle) < 5:
                        board.middle.append(card)
                    elif pos == "bottom" and len(board.bottom) < 5:
                        board.bottom.append(card)
            parent = child_node

        return Observation(
            board_self=board,
            board_opponent=root_obs.board_opponent,
            dealt_cards=dealt,
            known_discards_self=list(root_obs.known_discards_self),
            turn=root_obs.turn,
            is_btn=root_obs.is_btn,
            chips_self=root_obs.chips_self,
            chips_opponent=root_obs.chips_opponent,
        )

    def _backpropagate(self, node: MCTSNode, value: float):
        """Propagate value back up the tree."""
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent

    def _get_action_probs(self, root: MCTSNode) -> Dict[int, float]:
        """Convert visit counts to action probabilities."""
        visits = {idx: child.visit_count for idx, child in root.children.items()}
        total = sum(visits.values())
        if total == 0:
            n = len(visits)
            return {idx: 1.0 / n for idx in visits}

        if self.config.temperature == 0:
            # Greedy
            best = max(visits, key=visits.get)
            return {idx: (1.0 if idx == best else 0.0) for idx in visits}

        # Temperature-scaled
        temp = self.config.temperature
        scaled = {idx: (count ** (1.0 / temp)) for idx, count in visits.items()}
        total_scaled = sum(scaled.values())
        return {idx: v / total_scaled for idx, v in scaled.items()}

    def _select_action(self, action_probs: Dict[int, float]) -> int:
        """Sample action from probability distribution."""
        indices = list(action_probs.keys())
        probs = [action_probs[i] for i in indices]
        return random.choices(indices, weights=probs, k=1)[0]
